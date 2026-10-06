from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from fin_tracing.provider import build_exporter, build_tracer_provider


def test_build_exporter_falls_back_to_console_without_keys(monkeypatch):
    monkeypatch.delenv("ARIZE_API_KEY", raising=False)
    monkeypatch.delenv("ARIZE_SPACE_ID", raising=False)
    assert isinstance(build_exporter(), ConsoleSpanExporter)


def test_build_exporter_uses_otlp_with_keys(monkeypatch):
    monkeypatch.setenv("ARIZE_API_KEY", "key")
    monkeypatch.setenv("ARIZE_SPACE_ID", "space")
    assert isinstance(build_exporter(), OTLPSpanExporter)


def test_provider_resource_carries_project_and_service(monkeypatch):
    monkeypatch.setenv("ARIZE_PROJECT_NAME", "my-project")
    provider = build_tracer_provider("decision-agent", InMemorySpanExporter(), batch=False)
    attrs = provider.resource.attributes
    assert attrs["openinference.project.name"] == "my-project"
    assert attrs["service.name"] == "decision-agent"
