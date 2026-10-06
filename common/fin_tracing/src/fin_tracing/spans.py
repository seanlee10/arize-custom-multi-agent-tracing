"""OpenInference span helpers. Every span in the system is created through start_span()."""

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from openinference.instrumentation import get_attributes_from_context
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import baggage, trace
from opentelemetry.trace import Span, Status, StatusCode

TRACER_NAME = "fin_tracing"
# Used both as the baggage key that crosses processes and as the span attribute name.
TICKER_KEY = "finagent.ticker"


def serialize(value: Any) -> tuple[str, str]:
    """Return (text, mime type) for an input/output value."""
    if isinstance(value, str):
        return value, OpenInferenceMimeTypeValues.TEXT.value
    return json.dumps(value, default=str), OpenInferenceMimeTypeValues.JSON.value


class SpanHandle:
    def __init__(self, span: Span) -> None:
        self.span = span
        self.failed = False

    @property
    def trace_id(self) -> str:
        return format(self.span.get_span_context().trace_id, "032x")

    def set_output(self, value: Any) -> None:
        text, mime = serialize(value)
        self.span.set_attribute(SpanAttributes.OUTPUT_VALUE, text)
        self.span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, mime)

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        self.span.set_attributes(dict(attributes))

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        self.span.add_event(name, dict(attributes or {}))

    def fail(self, message: str) -> None:
        """Mark the span ERROR without raising (e.g. a rejected tool input)."""
        self.failed = True
        self.span.set_status(Status(StatusCode.ERROR, message))


@contextmanager
def start_span(
    name: str,
    kind: OpenInferenceSpanKindValues,
    *,
    input: Any = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[SpanHandle]:
    tracer = trace.get_tracer(TRACER_NAME)
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind.value)
        # No auto-instrumentor will copy session/user/metadata/tags/prompt template for us.
        span.set_attributes(dict(get_attributes_from_context()))
        ticker = baggage.get_baggage(TICKER_KEY)
        if ticker:
            span.set_attribute(TICKER_KEY, str(ticker))
        if input is not None:
            text, mime = serialize(input)
            span.set_attribute(SpanAttributes.INPUT_VALUE, text)
            span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, mime)
        if attributes:
            span.set_attributes(dict(attributes))
        handle = SpanHandle(span)
        try:
            yield handle
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
            raise
        if not handle.failed:
            span.set_status(Status(StatusCode.OK))


def agent_span(name: str, *, input: Any):
    return start_span(
        name, OpenInferenceSpanKindValues.AGENT, input=input, attributes={"finagent.agent.name": name}
    )


def chain_span(name: str, *, input: Any):
    return start_span(name, OpenInferenceSpanKindValues.CHAIN, input=input)


def tool_span(name: str, *, description: str, parameters_schema: dict[str, Any], arguments: dict[str, Any]):
    return start_span(
        name,
        OpenInferenceSpanKindValues.TOOL,
        input=arguments,
        attributes={
            SpanAttributes.TOOL_NAME: name,
            SpanAttributes.TOOL_DESCRIPTION: description,
            SpanAttributes.TOOL_PARAMETERS: json.dumps(parameters_schema),
        },
    )


def llm_span(name: str = "anthropic.messages"):
    return start_span(name, OpenInferenceSpanKindValues.LLM)
