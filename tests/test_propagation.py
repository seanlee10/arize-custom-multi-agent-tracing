import json

from fin_tracing.propagation import continue_request_context, inject_carrier, request_context
from fin_tracing.spans import chain_span
from tests.helpers import only_span


def test_carrier_continues_trace_and_request_attributes_in_fresh_context(exporter):
    with request_context(session_id="sess-1", user_id="user-1", request_id="req-1", ticker="AAPL"):
        with chain_span("caller", input="x"):
            carrier = inject_carrier()
    assert set(carrier) == {"traceparent", "baggage"}

    # The caller span has ended and its context is gone; only the carrier links the two.
    with continue_request_context(carrier):
        with chain_span("callee", input="y"):
            pass

    caller, callee = only_span(exporter, "caller"), only_span(exporter, "callee")
    assert callee.context.trace_id == caller.context.trace_id
    assert callee.parent.span_id == caller.context.span_id
    assert callee.attributes["session.id"] == "sess-1"
    assert callee.attributes["user.id"] == "user-1"
    assert callee.attributes["finagent.ticker"] == "AAPL"
    assert json.loads(callee.attributes["metadata"]) == {
        "request_id": "req-1",
        "ticker": "AAPL",
        "app_version": "0.1.0",
    }
    assert list(callee.attributes["tag.tags"]) == ["financial-analysis", "multi-agent", "demo"]


def test_header_names_are_case_insensitive(exporter):
    with request_context(session_id="s", user_id="u", request_id="r", ticker="MSFT"):
        with chain_span("caller", input="x"):
            carrier = {k.title(): v for k, v in inject_carrier().items()}
    with continue_request_context(carrier):
        with chain_span("callee", input="y"):
            pass
    assert only_span(exporter, "callee").parent.span_id == only_span(exporter, "caller").context.span_id


def test_missing_carrier_starts_new_trace_with_defaults(exporter):
    with continue_request_context({"io.modelcontextprotocol/clientInfo": {"name": "x"}}, ticker="NVDA"):
        with chain_span("orphan", input="x"):
            pass
    span = only_span(exporter, "orphan")
    assert span.parent is None
    assert span.attributes["user.id"] == "anonymous"
    assert span.attributes["finagent.ticker"] == "NVDA"
    assert span.attributes["session.id"]
