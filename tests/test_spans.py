import json

import pytest
from openinference.instrumentation import using_attributes
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import baggage
from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode

from fin_tracing.spans import TICKER_KEY, agent_span, chain_span, start_span, tool_span
from tests.helpers import only_span


def test_span_records_kind_io_and_ok_status(exporter):
    with agent_span("research_agent", input={"ticker": "AAPL"}) as span:
        span.set_output("done")
    s = only_span(exporter, "research_agent")
    assert s.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "AGENT"
    assert json.loads(s.attributes[SpanAttributes.INPUT_VALUE]) == {"ticker": "AAPL"}
    assert s.attributes[SpanAttributes.INPUT_MIME_TYPE] == "application/json"
    assert s.attributes[SpanAttributes.OUTPUT_VALUE] == "done"
    assert s.attributes[SpanAttributes.OUTPUT_MIME_TYPE] == "text/plain"
    assert s.attributes["finagent.agent.name"] == "research_agent"
    assert s.status.status_code is StatusCode.OK


def test_exception_marks_span_error_and_records_it(exporter):
    with pytest.raises(ValueError):
        with chain_span("step", input="x"):
            raise ValueError("boom")
    s = only_span(exporter, "step")
    assert s.status.status_code is StatusCode.ERROR
    assert "boom" in s.status.description
    assert [e.name for e in s.events] == ["exception"]


def test_fail_marks_error_without_raising(exporter):
    with start_span("soft", OpenInferenceSpanKindValues.TOOL) as span:
        span.fail("bad input")
    assert only_span(exporter, "soft").status.status_code is StatusCode.ERROR


def test_context_attributes_and_ticker_baggage_are_copied(exporter):
    token = otel_context.attach(baggage.set_baggage(TICKER_KEY, "MSFT"))
    try:
        with using_attributes(session_id="s-1", user_id="u-1", metadata={"k": "v"}, tags=["t"]):
            with chain_span("ctx", input="x"):
                pass
    finally:
        otel_context.detach(token)
    s = only_span(exporter, "ctx")
    assert s.attributes["session.id"] == "s-1"
    assert s.attributes["user.id"] == "u-1"
    assert json.loads(s.attributes["metadata"]) == {"k": "v"}
    assert list(s.attributes["tag.tags"]) == ["t"]
    assert s.attributes[TICKER_KEY] == "MSFT"


def test_tool_span_sets_tool_attributes(exporter):
    schema = {"type": "object", "properties": {"ticker": {"type": "string"}}}
    with tool_span("force_index", description="desc", parameters_schema=schema, arguments={"ticker": "AAPL"}):
        pass
    s = only_span(exporter, "force_index")
    assert s.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "TOOL"
    assert s.attributes[SpanAttributes.TOOL_NAME] == "force_index"
    assert s.attributes[SpanAttributes.TOOL_DESCRIPTION] == "desc"
    assert json.loads(s.attributes[SpanAttributes.TOOL_PARAMETERS]) == schema
    assert json.loads(s.attributes[SpanAttributes.INPUT_VALUE]) == {"ticker": "AAPL"}


def test_handle_exposes_trace_id(exporter):
    with chain_span("ids", input="x") as span:
        trace_id = span.trace_id
    assert trace_id == format(only_span(exporter, "ids").context.trace_id, "032x")
