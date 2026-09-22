import langgraph.version
import structlog
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import JSONResponse, PlainTextResponse

from langgraph_api import __version__, config, metadata
from langgraph_api.feature_flags import IS_POSTGRES_OR_GRPC_BACKEND
from langgraph_api.http_metrics import HTTP_METRICS_COLLECTOR
from langgraph_api.route import ApiRequest
from langgraph_api.schema import PoolStats, PostgresPoolStats, RedisPoolStats
from langgraph_runtime.database import connect, pool_stats
from langgraph_runtime.metrics import get_metrics

if IS_POSTGRES_OR_GRPC_BACKEND:
    from langgraph_api.grpc.ops import Runs
else:
    from langgraph_runtime.ops import Runs

METRICS_FORMATS = {"prometheus", "json"}

logger = structlog.stdlib.get_logger(__name__)


def _merge_pool_stats(local: PoolStats, remote: PoolStats) -> PoolStats:
    """Merge local and remote pool stats by summing numeric values. Used to aggregate Python + Go pool metrics."""
    merged: PoolStats = {}
    if "postgres" in local or "postgres" in remote:
        lp = local.get("postgres") or {}
        rp = remote.get("postgres") or {}
        merged["postgres"] = PostgresPoolStats(
            pool_max=lp.get("pool_max", 0) + rp.get("pool_max", 0),
            pool_size=lp.get("pool_size", 0) + rp.get("pool_size", 0),
            pool_available=lp.get("pool_available", 0) + rp.get("pool_available", 0),
            requests_queued=lp.get("requests_queued", 0) + rp.get("requests_queued", 0),
            requests_errors=lp.get("requests_errors", 0) + rp.get("requests_errors", 0),
        )
    if "redis" in local or "redis" in remote:
        lr = local.get("redis") or {}
        rr = remote.get("redis") or {}
        merged["redis"] = RedisPoolStats(
            idle_connections=lr.get("idle_connections", 0)
            + rr.get("idle_connections", 0),
            in_use_connections=lr.get("in_use_connections", 0)
            + rr.get("in_use_connections", 0),
            max_connections=lr.get("max_connections", 0) + rr.get("max_connections", 0),
        )
    return merged


async def _grpc_pool_stats() -> PoolStats:
    """Fetch connection pool stats from the Core API (Go) via gRPC for metrics aggregation. Returns {} on error."""
    if not IS_POSTGRES_OR_GRPC_BACKEND:
        return {}
    try:
        return await Runs.pool_stats()
    except Exception as e:
        await logger.awarning(
            "Failed to fetch Core API pool stats for aggregation", exc_info=e
        )
        return {}


async def meta_pool_stats() -> PoolStats:
    local_pool_stats: PoolStats = pool_stats()

    # Aggregate with Core API (Go) pool stats when using gRPC backend
    grpc_pool_stats = await _grpc_pool_stats()
    return _merge_pool_stats(local_pool_stats, grpc_pool_stats)


async def meta_info(request: ApiRequest):
    return JSONResponse(
        {
            "version": __version__,
            "langgraph_py_version": langgraph.version.__version__,
            "flags": {
                "assistants": True,
                "crons": True,
                "langsmith": bool(config.LANGSMITH_CONTROL_PLANE_API_KEY)
                and bool(config.TRACING),
                "langsmith_tracing_replicas": True,
                "langsmith_tracing_session_on_runs": True,
            },
            "host": {
                "kind": metadata.HOST,
                "project_id": metadata.PROJECT_ID,
                "host_revision_id": metadata.HOST_REVISION_ID,
                "revision_id": metadata.REVISION,
                "tenant_id": metadata.TENANT_ID,
            },
        }
    )


async def meta_metrics(request: ApiRequest):
    # determine output format
    metrics_format = request.query_params.get("format", "prometheus")
    if metrics_format not in METRICS_FORMATS:
        metrics_format = "prometheus"

    if metrics_format == "prometheus":
        # Served straight from the OTLP Prometheus client's registry (see
        # metrics_otlp._LSDPrometheusReader).
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # JSON: hand-built snapshot of workers, queue depth, HTTP, and pool stats.
    worker_metrics = get_metrics()["workers"]
    http_metrics = HTTP_METRICS_COLLECTOR.get_metrics(
        metadata.PROJECT_ID,
        metadata.HOST_REVISION_ID,
        metrics_format,
        metadata.DEPLOYMENT_TYPE,
    )
    merged_pool_stats = await meta_pool_stats()
    async with connect() as conn:
        resp = {
            **merged_pool_stats,
            "queue": await Runs.stats(conn),
            **http_metrics,
        }
        if config.N_JOBS_PER_WORKER > 0:
            resp["workers"] = worker_metrics
        return JSONResponse(resp)
