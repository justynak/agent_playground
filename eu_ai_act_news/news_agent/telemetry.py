"""OpenTelemetry wiring: traces, logs and metrics for the news agent pipeline.

This process is a short-lived batch job (one run per day), not a server, so the
usual "telemetry just runs in the background forever" assumption doesn't hold:
`setup_telemetry()` must be called once at the start of the run, and
`shutdown_telemetry()` once at the end (always, even on error/skip paths), or
the last run's spans/logs/metrics never get flushed out of the batch processors
before the process exits.

Everything is exported via OTLP/gRPC, configured entirely through the standard
`OTEL_EXPORTER_OTLP_*` environment variables (endpoint, headers, protocol —
see https://opentelemetry.io/docs/specs/otel/protocol/exporter/). With nothing
set, exporters default to `http://localhost:4317`, matching the local
docker-compose collector in `observability/`. No endpoint is hardcoded here:
pointing this at Grafana Cloud or any other OTLP-compatible backend is purely
an env var change, not a code change.

Modules acquire tracers/loggers/meters the normal OpenTelemetry way
(`trace.get_tracer(__name__)` etc.) regardless of whether `setup_telemetry()`
has run. Without it (e.g. in tests, or any import that isn't `main`), the
OpenTelemetry API falls back to its built-in no-op implementations, so
instrumented code is inert and side-effect-free until wired up.
"""

import logging

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = "eu-ai-act-news-agent"

# How long shutdown() waits to flush each signal before giving up. Keeps a run
# from hanging indefinitely if the collector is unreachable.
_FLUSH_TIMEOUT_MILLIS = 5_000

_tracer_provider: TracerProvider | None = None
_logger_provider: LoggerProvider | None = None
_meter_provider: MeterProvider | None = None
_log_handler: LoggingHandler | None = None

# One meter, and the instruments the pipeline records against. Created at
# import time against whatever meter provider is globally set at that point —
# safe even before `setup_telemetry()` runs, because `metrics.get_meter()`
# returns a proxy that forwards to the real provider once one is installed.
_meter = metrics.get_meter(__name__)

articles_collected_total = _meter.create_counter(
    "news_agent.articles.collected",
    unit="1",
    description="Relevant articles collected per feed, after keyword filtering (pre-dedup).",
)
articles_processed_total = _meter.create_counter(
    "news_agent.articles.processed",
    unit="1",
    description="Articles that finished the summarize/verify loop, labeled by outcome.",
)
verification_attempts_total = _meter.create_counter(
    "news_agent.verification.attempts",
    unit="1",
    description="Summarize+verify attempts, labeled by result (faithful/unfaithful/error).",
)
llm_call_duration_seconds = _meter.create_histogram(
    "news_agent.llm.call.duration",
    unit="s",
    description="DeepSeek chat completion latency, labeled by call type (generate/verify).",
)
digests_published_total = _meter.create_counter(
    "news_agent.digests.published",
    unit="1",
    description="Runs that ended in a published GitHub issue vs. a skip, labeled by reason.",
)


def setup_telemetry() -> None:
    """Configure and install the global trace/log/metric providers.

    Call exactly once, at the very start of the process. Reads endpoint/auth
    from `OTEL_EXPORTER_OTLP_*` env vars (see module docstring).
    """
    global _tracer_provider, _logger_provider, _meter_provider, _log_handler

    resource = Resource.create({"service.name": SERVICE_NAME})

    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(_tracer_provider)

    _logger_provider = LoggerProvider(resource=resource)
    _logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    _log_handler = LoggingHandler(level=logging.NOTSET, logger_provider=_logger_provider)
    logging.getLogger().addHandler(_log_handler)

    _meter_provider = MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
    )
    metrics.set_meter_provider(_meter_provider)


def shutdown_telemetry() -> None:
    """Flush and shut down all providers. Must be called before the process exits.

    Safe to call even if `setup_telemetry()` was never called (e.g. it failed
    partway through) — each provider is shut down only if it was created.
    """
    if _logger_provider is not None:
        logging.getLogger().removeHandler(_log_handler)
        _logger_provider.shutdown()
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
    if _meter_provider is not None:
        _meter_provider.shutdown(timeout_millis=_FLUSH_TIMEOUT_MILLIS)
