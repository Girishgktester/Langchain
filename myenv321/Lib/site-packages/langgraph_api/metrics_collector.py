"""Periodic collector that pushes snapshot/state metrics to the OTLP client.

This background task samples the same sources every ``STATS_INTERVAL_SECS`` and records them via the reporter.

The loop runs in **every** process (on postgres both the API server and the
dedicated queue worker share the same lifespan; inmem is a single process). Each
metric group self-gates so it lands on the right process:

- **worker gauges** — recorded wherever workers run (``N_JOBS_PER_WORKER > 0``):
  the queue worker, or a combined single-process deployment. A distributed API
  process (``N_JOBS_PER_WORKER == 0``) has no workers and skips them.
- **queue depth** (``num_pending_runs``/``num_running_runs``) — a single global
  value from ``Runs.stats`` (a gRPC call to the Go core). Emitted by the **API
  process only** (``not IS_QUEUE_ENTRYPOINT``) on the **postgres** runtime; inmem
  skips the DB round-trip entirely.
- **Postgres + Redis pool stats** — recorded on **both** processes (postgres
  runtime only), each reporting its own pools via ``meta_pool_stats()``, which
  merges the local Python pools with the Go-core pools. Redis stats in particular
  come from the local Python pool — the Go core omits them unless it has a
  non-cluster redis client — so a Go-core-only source would drop them. inmem has
  no real Postgres/Redis pools, so nothing is reported.

The two pool request counters are cumulative, so we push the delta since the
previous sample (OTLP counters are additive).

The loop also logs the same samples (``Worker stats``, ``Postgres pool stats``,
``Redis pool stats``) — folding in what the legacy per-process ``stats_loop``
functions used to log.
"""

from __future__ import annotations

import asyncio

import structlog

from langgraph_api import config
from langgraph_api.api.meta import meta_pool_stats
from langgraph_api.feature_flags import IS_POSTGRES_OR_GRPC_BACKEND
from langgraph_api.metrics_otlp import (
    COUNTER_PG_POOL_REQUESTS_ERRORS,
    COUNTER_PG_POOL_REQUESTS_QUEUED,
    GAUGE_NUM_PENDING_RUNS,
    GAUGE_NUM_RUNNING_RUNS,
    GAUGE_PG_POOL_AVAILABLE,
    GAUGE_PG_POOL_MAX,
    GAUGE_PG_POOL_SIZE,
    GAUGE_REDIS_POOL_AVAILABLE,
    GAUGE_REDIS_POOL_MAX,
    GAUGE_REDIS_POOL_SIZE,
    GAUGE_WORKERS_ACTIVE,
    GAUGE_WORKERS_AVAILABLE,
    GAUGE_WORKERS_MAX,
    get_otlp_metrics_reporter,
)
from langgraph_runtime.database import connect
from langgraph_runtime.metrics import get_metrics

if IS_POSTGRES_OR_GRPC_BACKEND:
    from langgraph_api.grpc.ops import Runs
else:
    from langgraph_runtime.ops import Runs

logger = structlog.stdlib.get_logger(__name__)


async def _collect_queue_and_workers(reporter) -> None:
    """Worker gauges (where workers run) + queue depth (API process only).

    Worker counts are local to the process running this loop and emitted wherever
    workers run (``N_JOBS_PER_WORKER > 0``) — the queue worker or a combined
    single-process deployment; a distributed API process (N_JOBS == 0) skips them.

    Queue depth is a single global value (from the run table, via ``Runs.stats`` —
    a gRPC call to the Go core) and is emitted by the **API process only**
    (``not IS_QUEUE_ENTRYPOINT``) on **postgres**: inmem skips the DB round-trip,
    and the dedicated queue worker leaves it to the API process so the global value
    is not double-reported across the queue/API split.
    """
    if config.N_JOBS_PER_WORKER > 0:
        workers = get_metrics()["workers"]
        reporter.record_gauge(GAUGE_WORKERS_MAX, workers["max"])
        reporter.record_gauge(GAUGE_WORKERS_ACTIVE, workers["active"])
        reporter.record_gauge(GAUGE_WORKERS_AVAILABLE, workers["available"])
        await logger.ainfo(
            "Worker stats",
            max=workers["max"],
            active=workers["active"],
            available=workers["available"],
        )

    # Queue depth is read from the run table via Runs.stats (a gRPC call to the
    # Go core on postgres). Emitted by the API process only — the queue worker
    # (IS_QUEUE_ENTRYPOINT) skips it. inmem skips the DB round-trip and reports
    # nothing.
    if IS_POSTGRES_OR_GRPC_BACKEND and not config.IS_QUEUE_ENTRYPOINT:
        async with connect() as conn:
            stats = await Runs.stats(conn)
        reporter.record_gauge(GAUGE_NUM_PENDING_RUNS, stats["n_pending"])
        reporter.record_gauge(GAUGE_NUM_RUNNING_RUNS, stats["n_running"])


async def _collect_pool(reporter, prev_counters: dict[str, int]) -> None:
    """Postgres + Redis pool gauges + cumulative request counters.

    Postgres runtime only, recorded on **both** processes (API server and queue
    worker) — each reports its own pools. Uses ``meta_pool_stats()``, which merges
    the local Python pools with the Go-core pools — matching the legacy /metrics.
    Redis stats come from the local Python pool (the Go core omits them unless it
    has a non-cluster redis client), so a Go-core-only source would drop them.
    """
    # Limitation: under BG_JOB_ISOLATED_LOOPS each worker runs in its own thread with its own
    # thread-local pg pool and redis client. This collector runs on the main
    # thread, so meta_pool_stats() -> _get_pool()/redis_stats() only sees the main
    # thread's pool (redis_stats() reads the global client unconditionally), and
    # the per-thread isolated pools are NOT aggregated — so pg/redis pool gauges
    # and the pg request counters under-report in isolated-loop mode.
    stats = await meta_pool_stats()

    pg = stats.get("postgres") or {}
    if pg:
        reporter.record_gauge(GAUGE_PG_POOL_MAX, pg.get("pool_max", 0))
        reporter.record_gauge(GAUGE_PG_POOL_SIZE, pg.get("pool_size", 0))
        reporter.record_gauge(GAUGE_PG_POOL_AVAILABLE, pg.get("pool_available", 0))
        # Cumulative counters: record the delta since the last sample. Emit on a
        # non-negative delta (>= 0) so the counter is created and reported from the
        # first sample even when it is 0 — the legacy /metrics always reported
        # these. Negative deltas (Go-core pool counter resets) are skipped to keep
        # the OTLP counter monotonic.
        for key, metric in (
            ("requests_queued", COUNTER_PG_POOL_REQUESTS_QUEUED),
            ("requests_errors", COUNTER_PG_POOL_REQUESTS_ERRORS),
        ):
            current = pg.get(key, 0)
            delta = current - prev_counters.get(key, 0)
            if delta >= 0:
                reporter.inc_counter(metric, delta)
            prev_counters[key] = current
        await logger.ainfo("Postgres pool stats", **pg)

    redis = stats.get("redis") or {}
    if redis:
        reporter.record_gauge(
            GAUGE_REDIS_POOL_AVAILABLE, redis.get("idle_connections", 0)
        )
        reporter.record_gauge(GAUGE_REDIS_POOL_SIZE, redis.get("in_use_connections", 0))
        reporter.record_gauge(GAUGE_REDIS_POOL_MAX, redis.get("max_connections", 0))
        await logger.ainfo("Redis pool stats", **redis)


async def _collect_once(prev_counters: dict[str, int]) -> None:
    reporter = get_otlp_metrics_reporter()
    if not reporter.enabled:
        return

    # Worker gauges are emitted wherever workers run; _collect_queue_and_workers
    # adds queue depth on the API process only (postgres; inmem skips the DB
    # round-trip).
    try:
        await _collect_queue_and_workers(reporter)
    except Exception as exc:
        await logger.awarning(
            "metrics collector: queue/worker sample failed", exc_info=exc
        )

    # Postgres/Redis pools live in the Go core (no real pools on inmem).
    if IS_POSTGRES_OR_GRPC_BACKEND:
        try:
            await _collect_pool(reporter, prev_counters)
        except Exception as exc:
            await logger.awarning("metrics collector: pool sample failed", exc_info=exc)


async def collector_loop() -> None:
    """Sample snapshot metrics into the OTLP client every STATS_INTERVAL_SECS."""
    interval = config.STATS_INTERVAL_SECS
    prev_counters: dict[str, int] = {}
    await logger.ainfo("Starting OTLP metrics collector loop", interval_secs=interval)
    try:
        while True:
            await _collect_once(prev_counters)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        pass
