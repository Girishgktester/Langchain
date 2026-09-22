from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import os
from collections.abc import Callable, Coroutine
from contextlib import ExitStack
from typing import TYPE_CHECKING, cast

import structlog
from langsmith import env as ls_env

from langgraph_runtime_inmem import database, ops

if TYPE_CHECKING:
    from langgraph_api.utils import future as lg_future

logger = structlog.stdlib.get_logger(__name__)

WORKERS: set[lg_future.AnyFuture] = set()

SHUTDOWN_GRACE_PERIOD_SECS = 5


class BgLoopRunner(asyncio.Runner):  # type: ignore[subclass-of-final-class]
    """A runner that runs a loop in a separate thread.

    Simplified version of the postgres BgLoopRunner — no connection pool
    or store cleanup needed for inmem.
    """

    executor: concurrent.futures.ThreadPoolExecutor

    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx

    def __enter__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            1, thread_name_prefix=f"bg-loop-{self.idx}"
        )
        self.executor.submit(self.get_loop).result()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        loop = self.get_loop()
        for task in asyncio.all_tasks(loop):
            task.cancel("Stopping background loop")
        self.executor.shutdown(wait=False)

    def submit(
        self,
        coro: Coroutine,
        *,
        name: str | None = None,
        callback: Callable[[lg_future.AnyFuture], None] | None = None,
    ):
        fut = self.executor.submit(self.run, coro, name=name)
        WORKERS.add(fut)
        if callback:
            fut.add_done_callback(callback)
        return fut

    def run(self, coro: Coroutine, *, name: str | None = None):
        """Run a coroutine inside the embedded event loop."""
        if asyncio.events._get_running_loop() is not None:
            raise RuntimeError(
                "Runner.run() cannot be called from a running event loop"
            )
        self._lazy_init()
        task = self._loop.create_task(coro, name=name)
        try:
            return self._loop.run_until_complete(task)
        except asyncio.exceptions.CancelledError:
            raise


def get_num_workers():
    return len(WORKERS)


async def queue():
    # Time and tide and asynchronous queues wait for no mortal,
    # As threads of our processes dance in delicate harmony,
    # Woven into the cosmic fabric of the server's eternal loom.
    # Imports delayed, like quantum particles, appearing only when observed.
    from langgraph_api import config, graph, webhook, worker  # noqa: PLC0415
    from langgraph_api.asyncio import AsyncQueue  # noqa: PLC0415

    concurrency = config.N_JOBS_PER_WORKER
    loop = asyncio.get_running_loop()
    last_stats_secs: int | None = None
    last_sweep_secs: int | None = None
    runners = AsyncQueue[BgLoopRunner](concurrency)
    WEBHOOKS: set[asyncio.Task] = set()
    # Not in public docs: dev-only, set by CLI --allow-blocking flag
    enable_blocking = os.getenv("LANGGRAPH_ALLOW_BLOCKING", "false").lower() == "true"
    # raise exceptions when a blocking call is detected inside an async function
    if enable_blocking:
        bb = None
        await logger.awarning(
            "Heads up: You've set --allow-blocking, which allows synchronous blocking I/O operations."
            " Be aware that blocking code in one run may tie up the shared event loop"
            " and slow down ALL other server operations. For best performance, use async drivers"
            " (e.g., aiohttp instead of requests, asyncpg instead of psycopg2). If switching to an"
            " async driver isn't possible, wrap the blocking call in asyncio.to_thread()."
        )
    else:
        bb = _enable_blockbuster()

    with ExitStack() as stack:
        if config.BG_JOB_ISOLATED_LOOPS:
            await logger.ainfo("Starting queue with isolated loops")
            executor = stack.enter_context(concurrent.futures.ThreadPoolExecutor())
            RUNNERS = {
                stack.enter_context(BgLoopRunner(idx)) for idx in range(concurrency)
            }
            for r in RUNNERS:
                runners.put_nowait(r)
                r.get_loop().set_default_executor(executor)
        else:
            await logger.ainfo("Starting queue with shared loop")
            for _ in range(concurrency):
                runners.put_nowait(cast(BgLoopRunner, object()))
        expired_runners: list[BgLoopRunner] = []

        def cleanup(
            task: lg_future.AnyFuture,
            runner: BgLoopRunner,
        ):
            WORKERS.discard(task)
            try:
                if config.BG_JOB_ISOLATED_LOOPS:
                    loop.call_soon_threadsafe(runners.put_nowait, runner)
                else:
                    runners.put_nowait(runner)
            except Exception as exc:
                expired_runners.append(runner)
                logger.exception("Background worker cleanup failed", exc_info=exc)

            try:
                if task.cancelled():
                    return
                if exc := task.exception():
                    if not isinstance(exc, asyncio.CancelledError):
                        logger.exception(
                            f"Background worker failed for task {task}",
                            exc_info=exc,
                        )
                    return
                result: worker.WorkerResult | None = task.result()
                if result and result["webhook"]:
                    if config.BG_JOB_ISOLATED_LOOPS:
                        hook_fut = asyncio.run_coroutine_threadsafe(
                            webhook.call_webhook(result), loop
                        )
                        WEBHOOKS.add(hook_fut)
                        hook_fut.add_done_callback(WEBHOOKS.remove)
                    else:
                        hook_task = loop.create_task(
                            webhook.call_webhook(result),
                            name=f"webhook-{result['run']['run_id']}",
                        )
                        WEBHOOKS.add(hook_task)
                        hook_task.add_done_callback(WEBHOOKS.remove)
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.exception("Background worker cleanup failed", exc_info=exc)

        await logger.ainfo(f"Starting {concurrency} background workers")
        try:
            run = None
            while True:
                if expired_runners:
                    await logger.awarning(
                        "Background worker expired, adding to queue",
                        num=len(expired_runners),
                    )
                    for runner in expired_runners:
                        await runners.put(runner)
                    expired_runners.clear()
                await runners.wait()
                try:
                    # check if we need to sweep runs
                    do_sweep = (
                        last_sweep_secs is None
                        or loop.time() - last_sweep_secs > config.BG_JOB_HEARTBEAT * 2
                    )
                    # check if we need to update stats
                    if calc_stats := (
                        last_stats_secs is None
                        or loop.time() - last_stats_secs > config.STATS_INTERVAL_SECS
                    ):
                        last_stats_secs = loop.time()
                        active = len(WORKERS)
                        await logger.ainfo(
                            "Worker stats",
                            max=concurrency,
                            available=concurrency - active,
                            active=active,
                        )
                    # skip the wait, if 1st time, or got a run last time
                    wait = run is None and last_stats_secs is not None
                    # try to get a run, handle it
                    run = None
                    async for run, attempt in ops.Runs.next(
                        wait=wait, limit=runners.qsize()
                    ):
                        runner = runners.get_nowait()
                        graph_id = (
                            run["kwargs"]
                            .get("config", {})
                            .get("configurable", {})
                            .get("graph_id")
                        )

                        if graph_id and graph.is_js_graph(graph_id):
                            task_name = f"js-run-{run['run_id']}-attempt-{attempt}"
                        else:
                            task_name = f"run-{run['run_id']}-attempt-{attempt}"

                        if not config.BG_JOB_ISOLATED_LOOPS or (
                            graph_id and graph.is_js_graph(graph_id)
                        ):
                            task = asyncio.create_task(
                                worker.worker(run, attempt, loop),
                                name=task_name,
                            )
                            task.add_done_callback(
                                functools.partial(cleanup, runner=runner)
                            )
                            WORKERS.add(task)
                        else:
                            runner.submit(
                                worker.worker(run, attempt, loop),
                                name=task_name,
                                callback=functools.partial(cleanup, runner=runner),
                            )
                    # run stats and sweep if needed
                    if calc_stats or do_sweep:
                        async with database.connect() as conn:
                            # update stats if needed
                            if calc_stats:
                                stats = await ops.Runs.stats(conn)
                                await logger.ainfo("Queue stats", **stats)
                            # sweep runs if needed
                            if do_sweep:
                                last_sweep_secs = loop.time()
                                await ops.Runs.sweep()
                except Exception as exc:
                    # keep trying to run the scheduler indefinitely
                    logger.exception("Background worker scheduler failed", exc_info=exc)
                    await exit.aclose()
        finally:
            if bb:
                bb.deactivate()
            logger.info("Shutting down background workers")
            for task in WORKERS:
                task.cancel()
            for task in WEBHOOKS:
                task.cancel()
            # When BG_JOB_ISOLATED_LOOPS is enabled, WORKERS and WEBHOOKS
            # contain concurrent.futures.Future objects which can't be
            # directly passed to asyncio.gather. Convert them first.
            from langgraph_api.utils.future import chain_future  # noqa: PLC0415

            futs: list[asyncio.Future] = []
            if config.BG_JOB_ISOLATED_LOOPS:
                futs.extend(
                    cast(
                        asyncio.Future,
                        chain_future(f, loop.create_future()),
                    )
                    for f in WORKERS
                )
                futs.extend(
                    cast(
                        asyncio.Future,
                        chain_future(f, loop.create_future()),
                    )
                    for f in WEBHOOKS
                )
            else:
                futs.extend(cast(asyncio.Future, f) for f in WORKERS)
                futs.extend(cast(asyncio.Future, f) for f in WEBHOOKS)
            await asyncio.wait_for(
                asyncio.gather(*futs, return_exceptions=True),
                SHUTDOWN_GRACE_PERIOD_SECS,
            )


def _enable_blockbuster():
    _patch_blocking_error()
    from blockbuster import BlockBuster  # noqa: PLC0415

    ls_env.get_runtime_environment()  # this gets cached
    bb = BlockBuster(excluded_modules=[])

    bb.functions["os.path.abspath"].can_block_in("inspect.py", "getmodule")
    for fn in (
        "os.access",
        "os.getcwd",
        "os.unlink",
        "os.write",
    ):
        bb.functions[fn].can_block_in(
            "langgraph_api/api/profile.py", "_profile_with_pyspy"
        )

    for module, func in (
        ("memory/__init__.py", "sync"),
        ("memory/__init__.py", "load"),
        ("memory/__init__.py", "dump"),
        ("pydantic/main.py", "__init__"),
    ):
        bb.functions["os.remove"].can_block_in(module, func)
        bb.functions["os.rename"].can_block_in(module, func)

    for module, func in (
        ("uvicorn/lifespan/on.py", "startup"),
        ("uvicorn/lifespan/on.py", "shutdown"),
        ("ansitowin32.py", "write_plain_text"),
        ("logging/__init__.py", "flush"),
        ("logging/__init__.py", "emit"),
    ):
        bb.functions["io.TextIOWrapper.write"].can_block_in(module, func)
        bb.functions["io.BufferedWriter.write"].can_block_in(module, func)
    # Support pdb
    bb.functions["builtins.input"].can_block_in("bdb.py", "trace_dispatch")
    bb.functions["builtins.input"].can_block_in("pdb.py", "user_line")
    to_disable = [
        "os.stat",
        # This is used by tiktoken for get_encoding_for_model
        # as well as importlib.metadata.
        "os.listdir",
        "os.remove",
        # We used to block the IO things but people use them so often that
        # we've decided to just let people make bad decisions for themselves.
        "io.BufferedReader.read",
        "io.BufferedWriter.write",
        "io.TextIOWrapper.read",
        "io.TextIOWrapper.write",
        # If people are using threadpoolexecutor, etc. they'd be using this.
        "threading.Lock.acquire",
    ]

    for function in bb.functions:
        if function.startswith("os.path."):
            to_disable.append(function)
    for function in to_disable:
        func = bb.functions.pop(function, None)
        if func:
            func.deactivate()
    bb.activate()

    return bb


def _patch_blocking_error():
    from blockbuster.blockbuster import BlockingError  # noqa: PLC0415

    original = BlockingError.__init__

    def init(self, func: str, *args, **kwargs):
        msg_ = func + (
            "\n\n"
            "Heads up! LangGraph dev identified a synchronous blocking call in your code. "
            "When running in an ASGI web server, blocking calls can degrade performance for everyone since they tie up the event loop.\n\n"
            "Here are your options to fix this:\n\n"
            "1. Best approach: Use an async driver so the call is non-blocking\n"
            "   For example, use 'await aiohttp.get()' instead of 'requests.get()', or asyncpg instead of psycopg2.\n\n"
            "2. If an async driver isn't available: wrap the blocking call in a thread\n"
            "   Example: 'await asyncio.to_thread(your_blocking_function)'\n\n"
            "3. Dev-only override: run 'langgraph dev --allow-blocking'\n\n"
            "These blocking operations can prevent health checks and slow down other runs in your deployment. "
            "Following these recommendations will help keep your LangGraph application running smoothly!"
        )
        original(self, msg_, *args, **kwargs)

    BlockingError.__init__ = init
