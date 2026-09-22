from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Literal

import structlog

from langgraph_api import __version__, config, metadata
from langgraph_api.http_metrics_utils import HTTP_LATENCY_BUCKETS

if TYPE_CHECKING:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.metrics import Observation
    from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
    from opentelemetry.sdk.metrics.export import (
        AggregationTemporality,
        MetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource
else:
    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.metrics import Observation
        from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
        from opentelemetry.sdk.metrics.export import (
            AggregationTemporality,
            MetricExporter,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource

        OTEL_AVAILABLE = True
    except ModuleNotFoundError:
        OTLPMetricExporter = None
        Observation = None
        MeterProvider = None
        PeriodicExportingMetricReader = None
        Resource = None
        AggregationTemporality = None
        Counter = object
        Histogram = object
        MetricExporter = object
        OTEL_AVAILABLE = False

    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader

        PROMETHEUS_EXPORTER_AVAILABLE = True
    except ModuleNotFoundError:
        # initialize as empty object to prevent breaking downstream inheritancei with _LSDPrometheusReader
        PrometheusMetricReader = object
        PROMETHEUS_EXPORTER_AVAILABLE = False

logger = structlog.stdlib.get_logger(__name__)

SERVICE_NAME = "lsd_langgraph_api"
DD_OTEL_METRIC_CONFIG = (
    '{"resource_attributes_as_tags":true,"histograms":{"mode":"distributions"}}'
)
METRIC_NAME_PREFIX = config.METRIC_PREFIX

METRIC_TIER_CRITICAL = 1
METRIC_TIER_INFO = 2
METRIC_TIER_DEBUG = 3
METRIC_TIER_DEEP_DEBUG = 4

# Legacy /metrics used HTTP_LATENCY_BUCKETS in seconds. OTLP record_latency()
# stores milliseconds, so convert (drop +Inf — OTel adds the overflow bucket).
# Apply to every latency metric: OTel defaults max out at 10000ms (~10s), which
# caps p95/p99 for long HTTP polls (/join), run queue waits (300s alerts), and
# run execution (30m alerts).
HTTP_LATENCY_BUCKETS_MS: tuple[float, ...] = tuple(
    b * 1000 for b in HTTP_LATENCY_BUCKETS if b != float("inf")
)

MetricType = Literal["counter", "histogram", "latency", "gauge"]


@dataclass(frozen=True, slots=True)
class MetricDef:
    metric_type: MetricType
    name: str
    tier: int
    # True for metrics surfaced on the LSD Deployment UI. This flag partitions the
    # two backends: the Prometheus scrape endpoint serves only these (see
    # _LSDPrometheusReader) so GCP indexes just the Deployment-UI metrics, while
    # Datadog gets only the internal complement (see _DatadogExporter).
    lsd_web_metric: bool = False
    # Human-readable help text. Passed to the OTel instrument as its description,
    # which the Prometheus exporter exposes as the metric's ``# HELP`` line.
    description: str = ""


def def_counter(
    name: str, tier: int, lsd_web_metric: bool = False, description: str = ""
) -> MetricDef:
    return MetricDef(
        metric_type="counter",
        name=f"{METRIC_NAME_PREFIX}{name}",
        tier=tier,
        lsd_web_metric=lsd_web_metric,
        description=description,
    )


def def_histogram(
    name: str, tier: int, lsd_web_metric: bool = False, description: str = ""
) -> MetricDef:
    return MetricDef(
        metric_type="histogram",
        name=f"{METRIC_NAME_PREFIX}{name}",
        tier=tier,
        lsd_web_metric=lsd_web_metric,
        description=description,
    )


def def_latency(
    name: str, tier: int, lsd_web_metric: bool = False, description: str = ""
) -> MetricDef:
    return MetricDef(
        metric_type="latency",
        name=f"{METRIC_NAME_PREFIX}{name}",
        tier=tier,
        lsd_web_metric=lsd_web_metric,
        description=description,
    )


def def_gauge(
    name: str, tier: int, lsd_web_metric: bool = False, description: str = ""
) -> MetricDef:
    return MetricDef(
        metric_type="gauge",
        name=f"{METRIC_NAME_PREFIX}{name}",
        tier=tier,
        lsd_web_metric=lsd_web_metric,
        description=description,
    )


# Pre-defined counter metrics.
COUNTER_STREAMING_DATA_LOSS = def_counter(
    "streaming_data_loss_counter", METRIC_TIER_CRITICAL
)
COUNTER_RUN_ATTEMPT_STARTED = def_counter(
    "run_attempt_started_counter", METRIC_TIER_CRITICAL
)
COUNTER_RUN_SUCCESS = def_counter("run_success_counter", METRIC_TIER_CRITICAL)
COUNTER_RUN_CANCELED_BY_REQUEST = def_counter(
    "run_canceled_by_request_counter", METRIC_TIER_CRITICAL
)
COUNTER_RUN_FAILED_RETRIABLE = def_counter(
    "run_failed_retriable_counter", METRIC_TIER_CRITICAL
)
COUNTER_RUN_FAILED_AFTER_RETRY = def_counter(
    "run_failed_after_retry_counter", METRIC_TIER_CRITICAL
)
COUNTER_RUN_EXCEED_MAX_ATTEMPTS_AT_START = def_counter(
    "run_exceed_max_attempts_at_start_counter", METRIC_TIER_CRITICAL
)
COUNTER_RUN_ABANDONED_BY_SHUTDOWN = def_counter(
    "run_abandoned_by_shutdown_counter", METRIC_TIER_CRITICAL
)
COUNTER_RUN_SET_STATUS_ERROR = def_counter(
    "run_set_status_error_counter", METRIC_TIER_CRITICAL
)
COUNTER_GRAPH_RECURSION_LIMIT_ERROR = def_counter(
    "graph_recursion_limit_error_counter", METRIC_TIER_INFO
)
COUNTER_FAILED_TO_FETCH_RUNS = def_counter(
    "failed_to_fetch_runs_counter", METRIC_TIER_CRITICAL
)
COUNTER_SERVER_STARTED = def_counter("server_started_counter", METRIC_TIER_INFO)
COUNTER_SERVER_REQUESTED_TO_STOP = def_counter(
    "server_requested_to_stop_counter", METRIC_TIER_INFO
)
COUNTER_SERVER_STOPPED = def_counter("server_stopped_counter", METRIC_TIER_INFO)
COUNTER_PROTOCOL_V2_BUFFER_EVICTED = def_counter(
    "protocol_v2_buffer_evicted_counter", METRIC_TIER_INFO
)
COUNTER_PROTOCOL_V2_EVENT_EMITTED = def_counter(
    "protocol_v2_event_emitted_counter", METRIC_TIER_DEBUG
)
COUNTER_PROTOCOL_V2_RESUME_GAP = def_counter(
    "protocol_v2_resume_gap_counter", METRIC_TIER_INFO
)
COUNTER_PROTOCOL_V2_TRANSPORT_SEND_FAILURE = def_counter(
    "protocol_v2_transport_send_failure_counter", METRIC_TIER_INFO
)
# Migrated from meta.py /metrics. Named to expose as `lg_api_http_requests_total`
# (this exporter version does not double-append `_total`).
COUNTER_HTTP_REQUESTS = def_counter(
    "http_requests_total", METRIC_TIER_INFO, lsd_web_metric=True
)
# Migrated from meta.py /metrics. The exporter appends `_total` to counter names,
# so these expose as `lg_api_pg_pool_requests_{queued,errors}_total` (idiomatic).
COUNTER_PG_POOL_REQUESTS_QUEUED = def_counter(
    "pg_pool_requests_queued",
    METRIC_TIER_CRITICAL,
    lsd_web_metric=True,
    description=(
        "Number of postgres connection requests queued because a postgres "
        "connection wasn't immediately available in the pool"
    ),
)
COUNTER_PG_POOL_REQUESTS_ERRORS = def_counter(
    "pg_pool_requests_errors",
    METRIC_TIER_CRITICAL,
    lsd_web_metric=True,
    description=(
        "Number of postgres connection requests resulting in an error "
        "(timeouts, queue full...)"
    ),
)

# Pre-defined latency metrics.
LATENCY_RUN_EXECUTION = def_latency("run_execution_latency", METRIC_TIER_INFO)
LATENCY_RUN_QUEUE_WAIT_TIME_1ST_ATTEMPT = def_latency(
    "run_queue_wait_time_1st_attempt",
    METRIC_TIER_INFO,
    lsd_web_metric=True,
    description=(
        "Time (milliseconds) spent by jobs waiting in the queue"
        " before getting processed for the first time. "
    ),
)
LATENCY_RUN_QUEUE_WAIT_TIME_RETRY_ATTEMPT = def_latency(
    "run_queue_wait_time_retry_attempt", METRIC_TIER_INFO
)
LATENCY_STREAM_PUBLISH = def_latency("stream_publish_latency", METRIC_TIER_INFO)
LATENCY_HTTP_REQUEST = def_latency(
    "http_requests_latency",
    METRIC_TIER_INFO,
    lsd_web_metric=True,
    description="HTTP request latency in milliseconds",
)

GAUGE_WORKERS_ACTIVE = def_gauge(
    "workers_active", METRIC_TIER_CRITICAL, lsd_web_metric=True
)
GAUGE_WORKERS_AVAILABLE = def_gauge(
    "workers_available", METRIC_TIER_CRITICAL, lsd_web_metric=True
)
GAUGE_PUBLISH_QUEUE_AVAILABILITY = def_gauge(
    "publish_queue_availability", METRIC_TIER_CRITICAL
)
# Snapshot/state gauges pushed by the periodic
# metrics collector loop (langgraph_api.metrics_collector).
# Queue depth + workers_max are inmem-only (the Go core emits them on postgres);
# pool stats are emitted on both runtimes.
GAUGE_WORKERS_MAX = def_gauge("workers_max", METRIC_TIER_CRITICAL, lsd_web_metric=True)
GAUGE_NUM_PENDING_RUNS = def_gauge(
    "num_pending_runs",
    METRIC_TIER_INFO,
    lsd_web_metric=True,
    description="The number of runs currently pending.",
)
GAUGE_NUM_RUNNING_RUNS = def_gauge(
    "num_running_runs",
    METRIC_TIER_INFO,
    lsd_web_metric=True,
    description="The number of runs currently running.",
)
GAUGE_PG_POOL_MAX = def_gauge(
    "pg_pool_max",
    METRIC_TIER_CRITICAL,
    lsd_web_metric=True,
    description="The maximum size of the postgres connection pool.",
)
GAUGE_PG_POOL_SIZE = def_gauge(
    "pg_pool_size",
    METRIC_TIER_CRITICAL,
    lsd_web_metric=True,
    description=(
        "Number of connections currently managed by the postgres connection "
        "pool (in the pool, given to clients, being prepared)"
    ),
)
GAUGE_PG_POOL_AVAILABLE = def_gauge(
    "pg_pool_available",
    METRIC_TIER_INFO,
    lsd_web_metric=True,
    description="Number of connections currently idle in the postgres connection pool",
)
GAUGE_REDIS_POOL_AVAILABLE = def_gauge(
    "redis_pool_available",
    METRIC_TIER_INFO,
    lsd_web_metric=True,
    description="Number of connections currently idle in the redis connection pool",
)
GAUGE_REDIS_POOL_SIZE = def_gauge(
    "redis_pool_size",
    METRIC_TIER_INFO,
    lsd_web_metric=True,
    description="Number of connections currently in use in the redis connection pool",
)
GAUGE_REDIS_POOL_MAX = def_gauge(
    "redis_pool_max",
    METRIC_TIER_INFO,
    lsd_web_metric=True,
    description="The maximum size of the redis connection pool.",
)
# Protocol v2 sessions retain a bounded replay buffer per run. Track the
# observed occupancy so operators can tune LSD_PROTOCOL_V2_BUFFER_SIZE before
# reconnects start seeing resume gaps.
GAUGE_PROTOCOL_V2_BUFFER_SIZE = def_gauge("protocol_v2_buffer_size", METRIC_TIER_DEBUG)


# Pre-defined histogram metrics.
HISTOGRAM_STREAM_DATA_SIZE = def_histogram("stream_data_size_bytes", METRIC_TIER_DEBUG)
HISTOGRAM_PROTOCOL_V2_REPLAYED_EVENTS = def_histogram(
    "protocol_v2_replayed_events", METRIC_TIER_DEBUG
)


# Names of metrics surfaced on the LSD Deployment UI. By default the two metric
# backends are partitioned by this set: the Prometheus scrape endpoint serves only
# these (see ``_LSDPrometheusReader``), and Datadog receives only the complement
# (see ``_DatadogExporter``). Setting EXPOSE_INTERNAL_METRICS_PROMETHEUS lifts the
# Prometheus filter so it serves every metric. Computed from the definitions above.
LSD_WEB_METRIC_NAMES: frozenset[str] = frozenset(
    m.name for m in globals().values() if isinstance(m, MetricDef) and m.lsd_web_metric
)


def _select_metrics(metrics_data: Any, keep) -> Any:
    """Return a copy of ``metrics_data`` keeping only metrics where ``keep(name)``.

    Rebuilt with ``dataclasses.replace`` (never mutated in place) since the two
    backend readers share the same SDK metric objects. Scope and resource groups
    left empty by the filter are dropped entirely, so a non-empty
    ``resource_metrics`` on the result guarantees at least one metric point.
    """
    if metrics_data is None or not metrics_data.resource_metrics:
        return metrics_data
    resource_metrics = []
    for rm in metrics_data.resource_metrics:
        scope_metrics = []
        for sm in rm.scope_metrics:
            kept = [m for m in sm.metrics if keep(m.name)]
            if kept:
                scope_metrics.append(replace(sm, metrics=kept))
        if scope_metrics:
            resource_metrics.append(replace(rm, scope_metrics=scope_metrics))
    return replace(metrics_data, resource_metrics=resource_metrics)


def _filter_web_metrics(metrics_data: Any) -> Any:
    """Copy of ``metrics_data`` with only LSD web metrics (for Prometheus/GCP)."""
    return _select_metrics(metrics_data, lambda name: name in LSD_WEB_METRIC_NAMES)


def _drop_web_metrics(metrics_data: Any) -> Any:
    """Copy of ``metrics_data`` without LSD web metrics (for Datadog)."""
    return _select_metrics(metrics_data, lambda name: name not in LSD_WEB_METRIC_NAMES)


class _LSDPrometheusReader(PrometheusMetricReader):
    """The Prometheus reader for this service.

    By default it serves only the LSD Deployment-UI (``lsd_web_metric``) set —
    Prometheus feeds the LSD web UI, while internal metrics go to Datadog instead
    (see ``_DatadogExporter``). When ``EXPOSE_INTERNAL_METRICS_PROMETHEUS`` is set,
    the web filter is skipped and every recorded metric is exposed (record-time
    tier filtering via ``METRIC_MAX_EMITTING_TIER`` still applies). The base
    ``_receive_metrics`` simply hands the data to its collector, so filtering here
    is sufficient.
    """

    def _receive_metrics(
        self,
        metrics_data: Any,
        timeout_millis: float = 10_000,
        **kwargs: Any,
    ) -> None:
        if not config.EXPOSE_INTERNAL_METRICS_PROMETHEUS:
            metrics_data = _filter_web_metrics(metrics_data)
        super()._receive_metrics(metrics_data, timeout_millis, **kwargs)


def _normalize_emitting_tier(value: int) -> int:
    if value < METRIC_TIER_CRITICAL:
        return METRIC_TIER_CRITICAL
    if value > METRIC_TIER_DEEP_DEBUG:
        return METRIC_TIER_DEEP_DEBUG
    return value


class _DatadogExporter(MetricExporter):
    """Datadog exporter wrapper: drops LSD web metrics, and skips export when
    nothing remains.

    Web metrics are served to the LSD Deployment UI via Prometheus only (see
    ``_LSDPrometheusReader``); Datadog receives only the internal complement. The
    drop happens here rather than at record time because the same SDK metric
    objects feed both backend readers.
    """

    def __init__(self, exporter: MetricExporter):
        super().__init__()
        self._exporter = exporter
        self._preferred_temporality = getattr(exporter, "_preferred_temporality", {})
        self._preferred_aggregation = getattr(exporter, "_preferred_aggregation", {})

    def export(
        self,
        metrics_data: Any,
        timeout_millis: float = 10_000,
        **kwargs: Any,
    ) -> Any:
        # _drop_web_metrics prunes emptied groups, so a non-empty resource_metrics
        # guarantees there is at least one internal metric point left to export.
        filtered = _drop_web_metrics(metrics_data)
        if not filtered or not filtered.resource_metrics:
            return None
        return self._exporter.export(filtered, timeout_millis, **kwargs)

    def shutdown(self, timeout_millis: float = 30_000, **kwargs: Any) -> None:
        self._exporter.shutdown(timeout_millis, **kwargs)

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        return self._exporter.force_flush(timeout_millis)


class OTelMetricsReporter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._initialized = False
        self._enabled = False
        self._meter_provider: MeterProvider | None = None
        self._meter = None
        self._max_tier = _normalize_emitting_tier(config.METRIC_MAX_EMITTING_TIER)
        self._instruments: dict[str, Any] = {}
        # Initializes a gauge values cache that is read when querying /metrics.
        #  sync gauges have a limitation of flapping values - it doesn't report the metric
        #  if the value wasn't set recently. By using a cache, it consistently reports the metric
        #  value when scraped.  Guarded by ``_gauge_lock`` because
        # callbacks run on the SDK collection thread, ``record_gauge`` on others.
        self._gauge_lock = threading.Lock()
        self._gauge_values: dict[str, dict[tuple, tuple[dict[str, Any], float]]] = {}
        self._observable_gauges: dict[str, Any] = {}
        # Labels attached to every metric (set in ``initialize``)
        self._common_attributes: dict[str, str] = {}
        self._prom_enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            self._initialized = True

            if not OTEL_AVAILABLE:
                logger.warning(
                    "OTel metrics disabled because OpenTelemetry dependencies are not installed"
                )
                return

            try:
                resource = Resource.create(
                    {
                        "service.name": SERVICE_NAME,
                        "k8s.pod.name": os.getenv("HOSTNAME", ""),
                        "api_version": os.getenv("LANGSMITH_LANGGRAPH_API_VERSION")
                        or __version__,
                        "project_id": os.getenv("LANGSMITH_HOST_PROJECT_ID", ""),
                        "k8s.deployment.name": os.getenv(
                            "LANGSMITH_HOST_PROJECT_NAME", ""
                        ),
                        "has_deepagents": "true"
                        if os.getenv("DEEPAGENTS_VERSION", "") != ""
                        else "false",
                        "deployment_type": os.getenv("LSD_DEPLOYMENT_TYPE", ""),
                    }
                )

                readers: list[Any] = []

                if config.DATADOG_METRICS_ENABLED:
                    base_exporter = OTLPMetricExporter(
                        endpoint=f"https://{config.LSD_DD_ENDPOINT}/v1/metrics",
                        headers={
                            "dd-api-key": config.LSD_DD_API_KEY,
                            "dd-otel-metric-config": DD_OTEL_METRIC_CONFIG,
                        },
                        preferred_temporality={
                            Counter: AggregationTemporality.DELTA,
                            Histogram: AggregationTemporality.DELTA,
                        },
                    )
                    # Burn after reading: remove DD keys from env so user code
                    # and child processes cannot access them.
                    for key in (
                        "LSD_DD_API_KEY",
                        "LSD_DD_ENDPOINT",
                    ):
                        os.environ.pop(key, None)

                    readers.append(
                        PeriodicExportingMetricReader(
                            _DatadogExporter(base_exporter),
                            export_interval_millis=10_000,
                        )
                    )

                # Prometheus metrics are always exported (served via /metrics).
                if not PROMETHEUS_EXPORTER_AVAILABLE:
                    logger.error(
                        "Prometheus metrics disabled: opentelemetry-exporter-prometheus not installed"
                    )
                else:
                    # PrometheusMetricReader registers its collector with the
                    # global prometheus_client REGISTRY, which the /metrics
                    # endpoint serves via generate_latest() (see api/meta.py).
                    # Prometheus serves only Deployment-UI metrics.
                    readers.append(_LSDPrometheusReader())
                    self._prom_enabled = True

                if not readers:
                    logger.info(
                        "OTel metrics reporter disabled (no backend configured)"
                    )
                    return

                self._meter_provider = MeterProvider(
                    resource=resource, metric_readers=readers
                )
                self._meter = self._meter_provider.get_meter(SERVICE_NAME)
                # Labels added to every metric, matching the legacy /metrics.
                self._common_attributes = {
                    "project_id": metadata.PROJECT_ID or "",
                    "revision_id": metadata.HOST_REVISION_ID or "",
                    "deployment_type": metadata.DEPLOYMENT_TYPE or "",
                }
                self._enabled = True

                if config.DATADOG_METRICS_ENABLED:
                    logger.info(
                        "Datadog OTLP metrics reader initialized",
                        endpoint=f"https://{config.LSD_DD_ENDPOINT}/v1/metrics",
                    )

                logger.info(
                    "OTel metrics reporter initialized",
                    metric_prefix=METRIC_NAME_PREFIX,
                    max_emitting_tier=self._max_tier,
                    backends=[
                        b
                        for b in (
                            "datadog" if config.DATADOG_METRICS_ENABLED else None,
                            "prometheus" if self._prom_enabled else None,
                        )
                        if b
                    ],
                )
            except Exception:
                self._enabled = False
                self._meter_provider = None
                self._meter = None
                logger.exception("Failed to initialize OTel metrics reporter")
                raise

    def shutdown(self) -> None:
        with self._lock:
            if self._meter_provider:
                try:
                    # Unregisters the Prometheus reader's collector from the global
                    # prometheus_client REGISTRY (and flushes/stops other readers).
                    self._meter_provider.shutdown()
                except Exception:
                    logger.exception("Failed to shutdown OTel metrics reporter")
                finally:
                    self._meter_provider = None
                    self._meter = None
            self._prom_enabled = False
            self._enabled = False
            self._initialized = False
            self._instruments.clear()
            with self._gauge_lock:
                self._gauge_values.clear()
                self._observable_gauges.clear()

    def _instrument_name(self, metric_name: str) -> str:
        return metric_name

    def _tier_enabled(self, tier: int) -> bool:
        return _normalize_emitting_tier(tier) <= self._max_tier

    def _should_emit(self, metric: MetricDef) -> bool:
        """Whether a sample for ``metric`` should be recorded.

        ``lsd_web_metric`` metrics bypass tier filtering: they back the LSD
        Deployment UI (served by the Prometheus reader) and must be emitted even
        on low-tier deployments (dev/dev_free default ``METRIC_MAX_EMITTING_TIER``
        to 1/CRITICAL). The tier gate runs before the MeterProvider, so a dropped
        sample never reaches any reader — Prometheus included.
        """
        if not self._enabled or not self._meter:
            return False
        return metric.lsd_web_metric or self._tier_enabled(metric.tier)

    def _get_or_create_instrument(self, metric: MetricDef):
        name = self._instrument_name(metric.name)
        instrument = self._instruments.get(name)
        if instrument is not None:
            return instrument
        if metric.metric_type == "counter":
            instrument = self._meter.create_counter(
                name=name, description=metric.description
            )
        elif metric.metric_type in {"histogram", "latency"}:
            # All latency metrics use legacy second-scale buckets (as ms).
            # Non-latency histograms (bytes, counts) keep OTel defaults.
            advisory = (
                list(HTTP_LATENCY_BUCKETS_MS)
                if metric.metric_type == "latency"
                else None
            )
            instrument = self._meter.create_histogram(
                name=name,
                description=metric.description,
                explicit_bucket_boundaries_advisory=advisory,
            )
        else:
            # Gauges are handled via observable instruments (see _set_gauge).
            raise ValueError(f"Unsupported metric type: {metric.metric_type}")
        self._instruments[name] = instrument
        return instrument

    def _make_gauge_callback(self, name: str):
        """Build the observable-gauge callback that the SDK invokes on each scrape.

        It yields one Observation per recorded attribute-set from the cache, so the
        last sampled value is re-reported on every collect (no flapping).
        """

        def _callback(_options: Any):
            with self._gauge_lock:
                points = list(self._gauge_values.get(name, {}).values())
            return [Observation(value, attributes=attrs) for attrs, value in points]

        return _callback

    def _with_common(self, attributes: dict[str, Any] | None) -> dict[str, Any]:
        """Merge the shared labels (project_id/revision_id/deployment_type) with
        any per-call attributes. Per-call values win on key conflicts."""
        return {**self._common_attributes, **(attributes or {})}

    def _set_gauge(self, metric: MetricDef, value: float, attributes: dict) -> None:
        name = metric.name
        key = tuple(sorted(attributes.items()))
        with self._gauge_lock:
            self._gauge_values.setdefault(name, {})[key] = (attributes, float(value))
            if name not in self._observable_gauges:
                self._observable_gauges[name] = self._meter.create_observable_gauge(
                    name=name,
                    description=metric.description,
                    callbacks=[self._make_gauge_callback(name)],
                )

    def inc_counter(
        self,
        metric: MetricDef,
        value: int = 1,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if metric.metric_type != "counter":
            raise ValueError(f"{metric.name} is not a counter metric")
        if not self._should_emit(metric):
            return
        instrument = self._get_or_create_instrument(metric)
        try:
            instrument.add(value, self._with_common(attributes))
        except Exception:
            logger.warning("Failed to add counter", metric_name=metric.name)

    def record_histogram(
        self,
        metric: MetricDef,
        value: float,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if metric.metric_type != "histogram":
            raise ValueError(f"{metric.name} is not a histogram metric")
        if not self._should_emit(metric):
            return
        instrument = self._get_or_create_instrument(metric)
        try:
            instrument.record(value, self._with_common(attributes))
        except Exception:
            logger.warning("Failed to record histogram", metric_name=metric.name)

    def record_latency(
        self,
        metric: MetricDef,
        duration_seconds: float | timedelta,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if metric.metric_type != "latency":
            raise ValueError(f"{metric.name} is not a latency metric")
        if isinstance(duration_seconds, timedelta):
            seconds = duration_seconds.total_seconds()
        else:
            seconds = float(duration_seconds)
        value = seconds * 1000
        if not self._should_emit(metric):
            return
        instrument = self._get_or_create_instrument(metric)
        try:
            instrument.record(value, self._with_common(attributes))
        except Exception:
            logger.warning("Failed to record latency", metric_name=metric.name)

    def record_gauge(
        self,
        metric: MetricDef,
        value: float,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        if metric.metric_type != "gauge":
            raise ValueError(f"{metric.name} is not a gauge metric")
        if not self._should_emit(metric):
            return
        try:
            # Cache the value; an observable gauge re-reports it on every scrape.
            self._set_gauge(metric, value, self._with_common(attributes))
        except Exception:
            logger.warning("Failed to record gauge", metric_name=metric.name)

    @contextmanager
    def track_latency_ms(
        self,
        metric: MetricDef,
        attributes: dict[str, Any] | None = None,
    ):
        if not self._should_emit(metric):
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record_latency(
                metric,
                time.perf_counter() - start,
                attributes=attributes,
            )


_metrics_reporter: OTelMetricsReporter | None = None


def get_otlp_metrics_reporter() -> OTelMetricsReporter:
    global _metrics_reporter
    if _metrics_reporter is None:
        _metrics_reporter = OTelMetricsReporter()
    return _metrics_reporter
