import httpx
import pytest
from opentelemetry.trace import StatusCode

from decision_agent.app import create_app as create_decision_app
from fin_tracing import request_context
from research_agent.decision_client import invoke_decision_agent
from research_agent.mcp_client import WebSearchClient
from tests.fakes import FakeAnthropic, fake_local_tool, message, tool_use
from tests.helpers import only_span
from websearch_mcp.server import create_server


async def fake_search(query: str, max_results: int) -> list[dict]:
    return [{"title": query, "url": "https://example.com", "content": "c"}]


def spans_with_side(exporter, side):
    return [s for s in exporter.get_finished_spans() if s.attributes.get("finagent.mcp.side") == side]


async def test_web_search_client_lists_tools_and_links_server_span(exporter):
    with request_context(session_id="s", user_id="u", request_id="r", ticker="AAPL"):
        async with WebSearchClient(create_server(fake_search)) as web_search:
            [definition] = web_search.tool_definitions()
            results = await web_search.call("web_search", {"query": "AAPL news"})

    assert definition["name"] == "web_search"
    assert definition["input_schema"]["required"] == ["query"]
    assert results[0]["title"] == "AAPL news"
    [client_span] = spans_with_side(exporter, "client")
    [server_span] = spans_with_side(exporter, "server")
    assert server_span.parent.span_id == client_span.context.span_id
    assert [e.name for e in client_span.events] == ["mcp.request_sent", "mcp.response_received"]


async def test_web_search_client_raises_on_tool_error(exporter):
    async def broken(query: str, max_results: int) -> list[dict]:
        raise RuntimeError("down")

    async with WebSearchClient(create_server(broken)) as web_search:
        with pytest.raises(RuntimeError, match="web_search failed"):
            await web_search.call("web_search", {"query": "x"})
    [client_span] = spans_with_side(exporter, "client")
    assert client_span.status.status_code is StatusCode.ERROR


async def test_invoke_decision_agent_propagates_trace(exporter):
    client = FakeAnthropic(
        [message(tool_use("tu", "submit_decision", {"decision": "hold", "confidence": 0.5, "rationale": "r"}), stop_reason="tool_use")]
    )
    app = create_decision_app(client=client, model="m", tools={"force_index": fake_local_tool("force_index", {})})
    with request_context(session_id="s", user_id="u", request_id="r", ticker="AAPL"):
        result = await invoke_decision_agent(
            "http://decision", ticker="AAPL", research_brief="b", transport=httpx.ASGITransport(app=app)
        )
    assert result["decision"] == "hold"
    chain, agent = only_span(exporter, "invoke_decision_agent"), only_span(exporter, "decision_agent")
    assert chain.attributes["openinference.span.kind"] == "CHAIN"
    assert agent.parent.span_id == chain.context.span_id


async def test_invoke_decision_agent_raises_on_5xx(exporter):
    transport = httpx.MockTransport(lambda request: httpx.Response(502, json={"detail": "x"}))
    with pytest.raises(httpx.HTTPStatusError):
        await invoke_decision_agent("http://decision", ticker="AAPL", research_brief="b", transport=transport)
    assert only_span(exporter, "invoke_decision_agent").status.status_code is StatusCode.ERROR
