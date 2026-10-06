import json

from mcp.client import Client
from opentelemetry.trace import StatusCode

from fin_tracing import chain_span, inject_carrier, request_context
from tests.helpers import only_span
from websearch_mcp.server import create_server


async def fake_search(query: str, max_results: int) -> list[dict]:
    return [{"title": f"T {query}", "url": "https://example.com", "content": "c"}][:max_results]


async def test_web_search_continues_callers_trace_via_meta(exporter):
    with request_context(session_id="s-1", user_id="u-1", request_id="r-1", ticker="AAPL"):
        with chain_span("caller", input="x"):
            carrier = inject_carrier()

    # Client is entered outside the caller's context: parenting can only come from _meta.
    async with Client(create_server(fake_search)) as client:
        result = await client.call_tool("web_search", {"query": "AAPL news", "max_results": 3}, meta=carrier)

    assert not result.is_error
    assert json.loads(result.content[0].text)[0]["title"] == "T AAPL news"
    tool, caller = only_span(exporter, "web_search"), only_span(exporter, "caller")
    assert tool.context.trace_id == caller.context.trace_id
    assert tool.parent.span_id == caller.context.span_id
    assert tool.attributes["openinference.span.kind"] == "TOOL"
    assert tool.attributes["session.id"] == "s-1"
    assert tool.attributes["finagent.ticker"] == "AAPL"
    assert tool.attributes["finagent.mcp.side"] == "server"
    assert tool.attributes["finagent.search.result_count"] == 1
    assert tool.status.status_code is StatusCode.OK


async def test_search_failure_returns_mcp_error_and_error_span(exporter):
    async def broken(query: str, max_results: int) -> list[dict]:
        raise RuntimeError("tavily down")

    async with Client(create_server(broken)) as client:
        result = await client.call_tool("web_search", {"query": "x"})

    assert result.is_error
    span = only_span(exporter, "web_search")
    assert span.status.status_code is StatusCode.ERROR
    assert "tavily down" in span.status.description
