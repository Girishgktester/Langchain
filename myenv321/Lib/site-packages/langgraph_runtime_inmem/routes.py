"""Internal routes for inmem runtime (testing/debugging only)."""

import asyncio
import collections
import json
import logging
import os
import re
import sys
import time
import uuid

from langgraph_runtime_inmem.database import connect

logger = logging.getLogger(__name__)

DEPLOY_TIMEOUT_SECONDS = 900  # 15 minutes

_ALLOWED_DEPLOY_ORIGINS = {
    "https://smith.langchain.com",
    "https://eu.smith.langchain.com",
    "https://dev.smith.langchain.com",
    "https://staging.smith.langchain.com",
}


def _is_allowed_deploy_origin(origin: str) -> bool:
    """Check whether *origin* is permitted to call deploy endpoints."""
    if not origin:
        return True
    if re.match(r"http://(localhost|127\.0\.0\.1)(:\d+)?$", origin):
        return True
    return origin in _ALLOWED_DEPLOY_ORIGINS


# Bumped on any non-backward-compatible change to the GET /deploy response
# shape or the SSE event taxonomy.
DEPLOY_PROTOCOL_VERSION = 1

# CLI --json event types forwarded verbatim as named SSE events. Anything
# else with a `message` field is collapsed onto the `log` channel.
_NAMED_EVENT_TYPES = frozenset({"step", "result", "upload_progress", "heartbeat"})


class DeployOperation:
    """Tracks a single ``langgraph deploy`` subprocess and its output."""

    __slots__ = (
        "operation_id",
        "status",
        "exit_code",
        "started_at",
        "finished_at",
        "events",
        "last_status_url_event",
        "last_result_event",
        "_proc",
        "_reader_task",
        "_stderr_task",
        "_stderr_tail",
        "_timeout_task",
        "_total_appended",
    )

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        started_at: float,
    ) -> None:
        self.operation_id: str = uuid.uuid4().hex[:12]
        self.status: str = "running"  # running | succeeded | failed
        self.exit_code: int | None = None
        self.started_at: float = started_at
        self.finished_at: float | None = None
        # Pre-encoded SSE events: each entry is (event_name_bytes, payload_dict)
        # ready to be yielded by the stream generator.
        self.events: collections.deque[tuple[bytes, dict]] = collections.deque(
            maxlen=2000
        )
        # Raw CLI events captured for replay via GET /deploy. Stored verbatim
        # so the inmem route doesn't need to track every field the CLI may
        # add to these events over time.
        self.last_status_url_event: dict | None = None
        self.last_result_event: dict | None = None
        # Stderr ring buffer. Hidden on success (stderr is mostly OS/runtime
        # noise like macOS's MallocStackLogging chatter); flushed into the
        # log channel as `level: error` on non-zero exit so Click messages
        # and Python tracebacks aren't lost behind a bare exit code.
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=50)
        self._proc = proc
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._timeout_task: asyncio.Task | None = None
        self._total_appended: int = 0


_current_op: DeployOperation | None = None


async def _drain_stderr(op: DeployOperation) -> None:
    """Read stderr into ``op._stderr_tail`` (see field docstring for usage)."""
    if op._proc.stderr is None:
        return
    try:
        async for raw_line in op._proc.stderr:
            line = raw_line.decode(errors="replace").rstrip()
            if line:
                op._stderr_tail.append(line)
    except Exception:
        # Stderr drainer is best-effort; never let it abort the deploy.
        logger.exception("Error draining deploy subprocess stderr")


async def _read_subprocess(op: DeployOperation) -> None:
    """Drain subprocess stdout into the SSE event ring buffer.

    The CLI is invoked with ``--json``, so each line is a JSON object with
    an ``event`` discriminator:

    * ``status_url`` / ``result`` events are also stashed verbatim on the
      operation so ``GET /deploy`` can replay them without the inmem route
      tracking each field the CLI may add over time.
    * Events listed in ``_NAMED_EVENT_TYPES`` are forwarded as named SSE
      events with their full payload.
    * Remaining events with a ``message`` are collapsed onto the ``log``
      channel as ``{message, level}``.
    * Non-JSON lines become ``log`` with ``level: "log"``.

    On non-zero exit, ``_stderr_tail`` is flushed onto the log channel.
    """
    # ``None`` means we never reached a terminal state (e.g. task cancelled
    # mid-read). In that case leave ``op.status`` as ``"running"`` so we
    # don't lie about a deploy that was actually torn down.
    final_status: str | None = None
    try:
        assert op._proc.stdout is not None
        async for raw_line in op._proc.stdout:
            line = raw_line.decode(errors="replace").rstrip()
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                op.events.append((b"log", {"message": line, "level": "log"}))
                op._total_appended += 1
                continue

            evt_type = event.get("event")

            # Stash raw payloads for GET /deploy replay.
            if evt_type == "status_url":
                op.last_status_url_event = event
            elif evt_type == "result":
                op.last_result_event = event

            if evt_type in _NAMED_EVENT_TYPES:
                # SSE event names are US-ASCII per spec; encode strictly so
                # any unexpected non-ASCII event type from the CLI surfaces
                # as a clear error rather than silently mangled bytes.
                op.events.append((evt_type.encode("ascii"), event))
                op._total_appended += 1
                continue

            msg = event.get("message", "")
            if msg:
                op.events.append((b"log", {"message": msg, "level": evt_type or "log"}))
                op._total_appended += 1

        exit_code = await op._proc.wait()
        op.exit_code = exit_code
        final_status = "succeeded" if exit_code == 0 else "failed"
    except Exception:
        final_status = "failed"
    finally:
        # Bounded wait so a slow drain doesn't hold up the terminal SSE event.
        if op._stderr_task is not None and not op._stderr_task.done():
            try:
                await asyncio.wait_for(op._stderr_task, timeout=2.0)
            except Exception:
                # asyncio.wait_for cancels the task on TimeoutError; on any
                # other exception the task is already finished.
                pass

        if final_status == "failed" and op._stderr_tail:
            for line in op._stderr_tail:
                op.events.append((b"log", {"message": line, "level": "error"}))
                op._total_appended += 1

        op.finished_at = time.time()
        # Assign terminal status LAST so the SSE generator (which polls
        # ``op.status != "running"`` to emit the terminal event and close)
        # never observes a non-running status before stderr lines and
        # finished_at have been populated.
        if final_status is not None:
            op.status = final_status
        if op._timeout_task and not op._timeout_task.done():
            op._timeout_task.cancel()


DEPLOY_TERMINATE_GRACE_SECONDS = 5.0


async def _timeout_watchdog(
    op: DeployOperation,
    timeout_seconds: float = DEPLOY_TIMEOUT_SECONDS,
    grace_seconds: float = DEPLOY_TERMINATE_GRACE_SECONDS,
) -> None:
    """Terminate the subprocess if it exceeds *timeout_seconds*.

    Sends SIGTERM first to allow the OS (and any subprocess like ``docker push``)
    to clean up file handles and in-flight requests, then escalates to SIGKILL
    after *grace_seconds* if the process hasn't exited.
    """
    await asyncio.sleep(timeout_seconds)
    if not (op.status == "running" and op._proc and op._proc.returncode is None):
        logger.debug(
            "Deploy operation %s finished before timeout (status=%s)",
            op.operation_id,
            op.status,
        )
        return

    msg = f"Deploy timed out after {timeout_seconds}s, terminating subprocess"
    logger.warning("Deploy operation %s: %s", op.operation_id, msg)
    # Surface the timeout to the SSE client so the user sees why the deploy
    # ended instead of just observing the process exit.
    op.events.append((b"log", {"message": msg, "level": "error"}))
    op._total_appended += 1
    op._proc.terminate()
    try:
        await asyncio.wait_for(op._proc.wait(), timeout=grace_seconds)
        return
    except TimeoutError:
        pass

    msg = f"Deploy did not exit within {grace_seconds}s after SIGTERM, killing"
    logger.warning("Deploy operation %s: %s", op.operation_id, msg)
    op.events.append((b"log", {"message": msg, "level": "error"}))
    op._total_appended += 1
    op._proc.kill()


def get_internal_routes():
    from langgraph_api.config import MIGRATIONS_PATH  # noqa: PLC0415

    try:
        from langgraph_api.middleware import http_logger  # noqa: PLC0415

        http_logger.PATHS_IGNORE.add("/internal/truncate")
    except ImportError:
        pass

    if "__inmem" not in MIGRATIONS_PATH:
        # not in a testing mode.
        return []
    from langgraph_api.route import ApiRequest, ApiResponse, ApiRoute  # noqa: PLC0415
    from langgraph_api.sse import EventSourceResponse  # noqa: PLC0415

    async def truncate(request: ApiRequest):
        """Truncate all inmem data (for testing)."""
        from langgraph_api import config as api_config  # noqa: PLC0415

        if api_config.USE_CUSTOM_CHECKPOINTER:
            from langgraph_api._checkpointer._adapter import (  # noqa: PLC0415
                CHECKPOINTER_STACK,
            )

            inner = getattr(CHECKPOINTER_STACK, "inner", None)
            if inner is not None and hasattr(inner, "clear"):
                await asyncio.to_thread(inner.clear)
        else:
            from langgraph_runtime.checkpoint import Checkpointer  # noqa: PLC0415

            await asyncio.to_thread(Checkpointer().clear)
        async with connect() as conn:
            await asyncio.to_thread(conn.clear)
        return ApiResponse({"ok": True})

    async def debug_get_raw_thread(request: ApiRequest):
        """Return raw thread from store without decryption (for testing)."""
        thread_id = request.path_params["thread_id"]
        async with connect() as conn:
            for thread in conn.store["threads"]:
                if str(thread["thread_id"]) == thread_id:
                    return ApiResponse(thread)
        return ApiResponse({"error": "not found"}, status_code=404)

    # Deploy endpoints

    def _check_deploy_origin(request: ApiRequest):
        origin = request.headers.get("origin", "")
        if not _is_allowed_deploy_origin(origin):
            return ApiResponse({"error": "origin not allowed"}, status_code=403)
        return None

    def _collect_deploy_info() -> dict:
        """Gather deploy metadata (runs in a thread to avoid blocking the loop)."""
        cwd = os.getcwd()
        return {
            "protocol_version": DEPLOY_PROTOCOL_VERSION,
            "default_name": re.sub(r"[^a-z0-9-]", "-", os.path.basename(cwd).lower()),
            "has_api_key": bool(
                os.environ.get("LANGSMITH_API_KEY")
                or os.environ.get("LANGCHAIN_API_KEY")
                or os.environ.get(
                    "LANGGRAPH_HOST_API_KEY"
                )  # Not in public docs: internal
            ),
            "has_config": os.path.isfile(os.path.join(cwd, "langgraph.json")),
        }

    def _serialize_operation(op: DeployOperation) -> dict:
        # Raw passthrough so new CLI fields propagate without server changes.
        return {
            "operation_id": op.operation_id,
            "status": op.status,
            "exit_code": op.exit_code,
            "started_at": op.started_at,
            "finished_at": op.finished_at,
            "status_url_event": op.last_status_url_event,
            "result_event": op.last_result_event,
        }

    async def get_deploy(request: ApiRequest):
        """Return deploy metadata and current operation state (if any).

        Combines what was previously /deploy/info and /deploy/active into
        a single GET /deploy endpoint to reduce round-trips.
        """
        if err := _check_deploy_origin(request):
            return err
        info = await asyncio.to_thread(_collect_deploy_info)
        payload: dict = {**info, "current_operation": None}
        if _current_op:
            payload["current_operation"] = _serialize_operation(_current_op)
        return ApiResponse(payload)

    async def start_deploy(request: ApiRequest):
        """Start a new deploy operation."""
        global _current_op

        if err := _check_deploy_origin(request):
            return err

        if _current_op and _current_op.status == "running":
            return ApiResponse(
                {
                    "error": "Deploy already in progress",
                    "operation_id": _current_op.operation_id,
                },
                status_code=409,
            )

        try:
            body = await request.json()
        except Exception:
            body = {}
        name = body.get("name") if isinstance(body, dict) else None

        cmd = [
            sys.executable,
            "-m",
            "langgraph_cli",
            "deploy",
            "--json",
            "--no-input",
        ]
        if name:
            cmd.extend(["--name", str(name)])

        cwd = await asyncio.to_thread(os.getcwd)
        env = {**os.environ, "LANGGRAPH_CLI_ANALYTICS_SOURCE": "studio"}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        op = DeployOperation(proc=proc, started_at=time.time())
        op._reader_task = asyncio.create_task(_read_subprocess(op))
        op._stderr_task = asyncio.create_task(_drain_stderr(op))
        op._timeout_task = asyncio.create_task(_timeout_watchdog(op))
        _current_op = op

        return ApiResponse({"operation_id": op.operation_id, "status": op.status})

    async def stream_deploy_logs(request: ApiRequest):
        """SSE stream of buffered deploy events.
        ---
        summary: SSE stream of buffered deploy events.
        description: |
            Each event carries a monotonic ``id``; clients can resume after a
            disconnect by passing it back via the standard ``Last-Event-ID``
            header (sent automatically by ``EventSource`` on auto-reconnect) or
            a ``last_event_id`` query parameter (for explicit resume on a fresh
            connection, since browsers don't expose a way to set the header on
            ``new EventSource``).

            Events on this stream:

            * ``log`` — ``{message, level}``. ``level`` is one of ``log``, ``info``,
              ``warn``, ``note``, ``error``, ``status_change``, ``status_url``
              (CLI-side severity), letting the frontend color or icon lines.
            * ``step`` — ``{event, step, message, ...}`` from the CLI.
            * ``result`` — ``{event, status, deployment_id, message, url?, status_url?}``
              carrying the final outcome and dashboard/app URLs without a
              follow-up GET.
            * ``upload_progress`` — ``{event, size_mb, pct}`` during source archive
              upload.
            * ``done`` / ``error`` — terminal markers with ``{exit_code}``.
        """
        if err := _check_deploy_origin(request):
            return err
        op_id = request.path_params["operation_id"]
        if not _current_op or _current_op.operation_id != op_id:
            return ApiResponse({"error": "not found"}, status_code=404)
        op = _current_op

        raw_cursor = request.headers.get("last-event-id") or request.query_params.get(
            "last_event_id"
        )
        try:
            initial_cursor = max(0, int(raw_cursor)) if raw_cursor else 0
        except (TypeError, ValueError):
            initial_cursor = 0

        async def generate():
            # Cursor is the absolute (1-indexed) id of the last event the
            # client has already seen. We resume from cursor+1. If the
            # cursor points before the deque start (buffer rotated past it),
            # we start from the earliest available — the gap is unrecoverable.
            cursor = initial_cursor

            while True:
                snapshot = list(op.events)
                total = op._total_appended
                deque_start = total - len(snapshot)
                skip = max(0, cursor - deque_start)
                for offset, (event_name, payload) in enumerate(snapshot[skip:]):
                    event_id = str(deque_start + skip + offset + 1).encode()
                    yield (event_name, payload, event_id)
                cursor = total

                if op.status != "running":
                    snapshot = list(op.events)
                    total = op._total_appended
                    deque_start = total - len(snapshot)
                    skip = max(0, cursor - deque_start)
                    for offset, (event_name, payload) in enumerate(snapshot[skip:]):
                        event_id = str(deque_start + skip + offset + 1).encode()
                        yield (event_name, payload, event_id)
                    terminal = b"done" if op.status == "succeeded" else b"error"
                    terminal_id = str(op._total_appended + 1).encode()
                    yield (terminal, {"exit_code": op.exit_code}, terminal_id)
                    return

                await asyncio.sleep(0.1)

        return EventSourceResponse(generate())

    # Routes are inserted via `unshadowable_meta_routes.insert(0, route)`
    # in api/__init__.py, which reverses the order.  Put parameterised
    # paths first so they end up LAST after reversal.
    return [
        ApiRoute(
            "/deploy/{operation_id}/stream",
            stream_deploy_logs,
            methods=["GET"],
        ),
        ApiRoute("/deploy", start_deploy, methods=["POST"]),
        ApiRoute("/deploy", get_deploy, methods=["GET"]),
        ApiRoute(
            "/internal/debug/thread/{thread_id}",
            debug_get_raw_thread,
            methods=["GET"],
        ),
        ApiRoute("/internal/truncate", truncate, methods=["POST"]),
    ]
