"""TracerProvider, exporter and propagator setup for exporting spans to Arize AX."""

import os

from openinference.semconv.resource import ResourceAttributes
from opentelemetry import trace
from opentelemetry.baggage.propagation import W3CBaggagePropagator
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import NoOpTracer, Tracer
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from fin_tracing.spans import TRACER_NAME

DEFAULT_PROJECT_NAME = "financial-multi-agent"
DEFAULT_OTLP_ENDPOINT = "https://otlp.arize.com/v1"


class FinTracingOnlyProvider(TracerProvider):
    """Records only spans created by fin_tracing.

    Libraries such as FastAPI and the MCP SDK emit their own OpenTelemetry spans as soon as a
    global provider exists. This sample is manual-only, so every other tracer gets a NoOpTracer:
    it records nothing but passes the parent context through, so our spans stay linked.
    """

    def get_tracer(self, instrumenting_module_name: str, *args, **kwargs) -> Tracer:
        if instrumenting_module_name != TRACER_NAME:
            return NoOpTracer()
        return super().get_tracer(instrumenting_module_name, *args, **kwargs)


def build_tracer_provider(
    service_name: str,
    exporter: SpanExporter,
    *,
    project_name: str | None = None,
    batch: bool = True,
) -> TracerProvider:
    resource = Resource.create(
        {
            ResourceAttributes.PROJECT_NAME: project_name
            or os.getenv("ARIZE_PROJECT_NAME", DEFAULT_PROJECT_NAME),
            "service.name": service_name,
        }
    )
    provider = FinTracingOnlyProvider(resource=resource)
    processor = BatchSpanProcessor(exporter) if batch else SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    return provider


def build_exporter() -> SpanExporter:
    """OTLP/gRPC exporter to Arize, or the console when Arize credentials are missing."""
    api_key = os.getenv("ARIZE_API_KEY")
    space_id = os.getenv("ARIZE_SPACE_ID")
    if not (api_key and space_id):
        return ConsoleSpanExporter()
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    return OTLPSpanExporter(
        endpoint=os.getenv("ARIZE_OTLP_ENDPOINT", DEFAULT_OTLP_ENDPOINT),
        headers={"space_id": space_id, "api_key": api_key},
    )


def configure_propagator() -> None:
    """traceparent links spans across processes; baggage carries session/user/request/ticker."""
    set_global_textmap(
        CompositePropagator([TraceContextTextMapPropagator(), W3CBaggagePropagator()])
    )


def setup_tracing(service_name: str) -> TracerProvider:
    provider = build_tracer_provider(service_name, build_exporter())
    trace.set_tracer_provider(provider)
    configure_propagator()
    return provider
