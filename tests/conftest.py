import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from fin_tracing.provider import build_tracer_provider, configure_propagator

# The global tracer provider can only be set once per process, so tests share one
# in-memory exporter and clear it before every test.
_EXPORTER = InMemorySpanExporter()
trace.set_tracer_provider(build_tracer_provider("tests", _EXPORTER, batch=False))
configure_propagator()


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    _EXPORTER.clear()
    return _EXPORTER
