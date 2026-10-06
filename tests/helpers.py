from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def spans_named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


def only_span(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    spans = spans_named(exporter, name)
    assert len(spans) == 1, f"expected one span named {name!r}, got {len(spans)}"
    return spans[0]
