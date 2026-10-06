import httpx
from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import StatusCode

from decision_agent.app import create_app as create_decision_app
from research_agent.agent import ResearchDeps
from research_agent.app import create_app as create_research_app
from tests.fakes import FakeAnthropic, fake_local_tool, message, text, tool_use
from tests.helpers import only_span
from websearch_mcp.server import create_server


async def fake_search(query: str, max_results: int) -> list[dict]:
    return [{"title": "Apple beats estimates", "url": "https://example.com", "content": "..."}]


async def test_one_request_produces_one_connected_trace(exporter):
    research_llm = FakeAnthropic(
        [
            message(tool_use("tu_1", "web_search", {"query": "AAPL earnings"}), stop_reason="tool_use"),
            message(text("Summary: solid quarter.")),
        ]
    )
    decision_llm = FakeAnthropic(
        [
            message(tool_use("tu_a", "force_index", {"ticker": "AAPL"}), stop_reason="tool_use"),
            message(
                tool_use("tu_b", "submit_decision", {"decision": "buy", "confidence": 0.7, "rationale": "FI bullish"}),
                stop_reason="tool_use",
            ),
        ]
    )
    decision_app = create_decision_app(
        client=decision_llm, model="claude-opus-5-5", tools={"force_index": fake_local_tool("force_index", {"trend": "bullish"})}
    )
    research_app = create_research_app(
        ResearchDeps(
            client=research_llm,
            model="claude-opus-5-5",
            websearch_server=create_server(fake_search),
            decision_url="http://decision-agent",
            decision_transport=httpx.ASGITransport(app=decision_app),
        )
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=research_app), base_url="http://research") as http:
        response = await http.post("/invoke", json={"ticker": "aapl", "session_id": "sess-42", "user_id": "analyst-1"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["decision"] == "buy"

    spans = exporter.get_finished_spans()
    # One trace for everything.
    assert {s.context.trace_id for s in spans} == {int(body["trace_id"], 16)}
    # Same session / user / ticker on every span, in every "container".
    for s in spans:
        assert s.attributes["session.id"] == "sess-42", s.name
        assert s.attributes["user.id"] == "analyst-1", s.name
        assert s.attributes["finagent.ticker"] == "AAPL", s.name
        assert s.status.status_code is StatusCode.OK, s.name

    root = only_span(exporter, "research_agent")
    assert root.parent is None
    [client_ws] = [s for s in spans if s.attributes.get("finagent.mcp.side") == "client"]
    [server_ws] = [s for s in spans if s.attributes.get("finagent.mcp.side") == "server"]
    assert client_ws.parent.span_id == root.context.span_id
    assert server_ws.parent.span_id == client_ws.context.span_id  # via MCP _meta
    chain = only_span(exporter, "invoke_decision_agent")
    decision = only_span(exporter, "decision_agent")
    assert chain.parent.span_id == root.context.span_id
    assert decision.parent.span_id == chain.context.span_id  # via HTTP headers
    assert only_span(exporter, "force_index").parent.span_id == decision.context.span_id

    kinds = [s.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] for s in spans]
    assert kinds.count("LLM") == 4
    assert kinds.count("AGENT") == 2
