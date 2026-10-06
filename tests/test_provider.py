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


def test_library_tracers_are_silenced_but_keep_parentage(exporter):
    from opentelemetry import trace

    from fin_tracing.spans import chain_span
    from tests.helpers import only_span

    library = trace.get_tracer("fastapi")  # FastAPI / MCP SDK emit their own spans
    with library.start_as_current_span("POST /invoke"):
        with chain_span("outer", input="x"):
            with library.start_as_current_span("tools/call"):
                with chain_span("inner", input="y"):
                    pass
    assert [s.name for s in exporter.get_finished_spans()] == ["inner", "outer"]
    assert only_span(exporter, "outer").parent is None
    assert only_span(exporter, "inner").parent.span_id == only_span(exporter, "outer").context.span_id
