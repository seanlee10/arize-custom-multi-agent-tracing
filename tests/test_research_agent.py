import httpx
from opentelemetry.trace import StatusCode

from research_agent.agent import ResearchDeps
from research_agent.app import create_app
from tests.fakes import FakeAnthropic, message, text, tool_use
from tests.helpers import only_span
from websearch_mcp.server import create_server


async def fake_search(query: str, max_results: int) -> list[dict]:
    return [{"title": query, "url": "https://example.com", "content": "c"}]


def research_llm() -> FakeAnthropic:
    return FakeAnthropic(
        [
            message(tool_use("tu_1", "web_search", {"query": "AAPL earnings"}), stop_reason="tool_use"),
            message(text("Summary: strong quarter.")),
        ]
    )


def decision_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"decision": "buy", "confidence": 0.7, "rationale": "r", "indicators": {}})


def make_app(llm: FakeAnthropic, decision_handler) -> object:
    return create_app(
        ResearchDeps(
            client=llm,
            model="claude-opus-5-5",
            websearch_server=create_server(fake_search),
            decision_url="http://decision",
            decision_transport=httpx.MockTransport(decision_handler),
        )
    )


async def post(app, payload):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://research") as http:
        return await http.post("/invoke", json=payload)


async def test_invoke_researches_then_returns_decision(exporter):
    llm = research_llm()
    response = await post(make_app(llm, decision_ok), {"ticker": "AAPL", "session_id": "sess-1"})
    body = response.json()
    assert response.status_code == 200
    assert body["decision"] == "buy"
    assert body["research_brief"] == "Summary: strong quarter."
    assert llm.calls[0]["tools"][0]["name"] == "web_search"  # discovered from the MCP server
    root = only_span(exporter, "research_agent")
    assert root.parent is None
    assert format(root.context.trace_id, "032x") == body["trace_id"]
    assert root.attributes["finagent.decision"] == "buy"
    assert root.attributes["session.id"] == "sess-1"
    assert root.attributes["user.id"] == "demo-user"


async def test_invoke_normalises_and_validates_ticker(exporter):
    llm = research_llm()
    ok = await post(make_app(llm, decision_ok), {"ticker": " aapl "})
    assert ok.status_code == 200
    assert ok.json()["ticker"] == "AAPL"
    bad_llm = FakeAnthropic([])
    bad = await post(make_app(bad_llm, decision_ok), {"ticker": "AAPL; rm -rf"})
    assert bad.status_code == 422
    assert bad_llm.calls == []


async def test_invoke_returns_502_with_trace_id_when_decision_agent_fails(exporter):
    response = await post(
        make_app(research_llm(), lambda request: httpx.Response(503, text="down")), {"ticker": "AAPL"}
    )
    assert response.status_code == 502
    detail = response.json()["detail"]
    root = only_span(exporter, "research_agent")
    assert detail["trace_id"] == format(root.context.trace_id, "032x")
    assert root.status.status_code is StatusCode.ERROR
    assert only_span(exporter, "invoke_decision_agent").status.status_code is StatusCode.ERROR


async def test_502_reports_the_underlying_error_not_an_exception_group(exporter):
    # The MCP client runs inside an anyio task group, which wraps errors in ExceptionGroup.
    response = await post(make_app(FakeAnthropic([]), decision_ok), {"ticker": "AAPL"})
    assert response.status_code == 502
    error = response.json()["detail"]["error"]
    assert "ran out of scripted responses" in error
    assert "ExceptionGroup" not in error
    assert "ExceptionGroup" not in only_span(exporter, "research_agent").status.description
