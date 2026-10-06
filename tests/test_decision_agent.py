import httpx
import pytest
from opentelemetry.trace import StatusCode

from decision_agent.agent import MAX_ITERATIONS, DecisionAgentError, run_decision_agent
from decision_agent.app import create_app
from fin_tracing import chain_span, inject_carrier, request_context
from tests.fakes import FakeAnthropic, fake_local_tool, message, text, tool_use
from tests.helpers import only_span, spans_named

TOOLS = {
    "force_index": fake_local_tool("force_index", {"trend": "bullish"}),
    "bollinger_bands": fake_local_tool("bollinger_bands", {"signal": "neutral"}),
    "insider_trades": fake_local_tool("insider_trades", {"net_shares": 10}),
}


def submit(decision="buy", confidence=0.8, rationale="Bullish FI", id="tu_s"):
    return tool_use(id, "submit_decision", {"decision": decision, "confidence": confidence, "rationale": rationale})


async def test_happy_path_runs_tools_then_submits(exporter):
    client = FakeAnthropic(
        [
            message(
                tool_use("tu_1", "force_index", {"ticker": "AAPL"}),
                tool_use("tu_2", "bollinger_bands", {"ticker": "AAPL"}),
                tool_use("tu_3", "insider_trades", {"ticker": "AAPL"}),
                stop_reason="tool_use",
            ),
            message(submit(), stop_reason="tool_use"),
        ]
    )
    result = await run_decision_agent("AAPL", "brief", client=client, model="claude-opus-5-5", tools=TOOLS)

    assert result["decision"] == "buy"
    assert result["confidence"] == 0.8
    assert set(result["indicators"]) == {"force_index", "bollinger_bands", "insider_trades"}
    # All three tool results go back in ONE user message.
    second_call_messages = client.calls[1]["messages"]
    assert [b["tool_use_id"] for b in second_call_messages[-1]["content"]] == ["tu_1", "tu_2", "tu_3"]
    submit_tool = next(t for t in client.calls[0]["tools"] if t["name"] == "submit_decision")
    assert submit_tool["strict"] is True
    assert submit_tool["input_schema"]["properties"]["decision"]["enum"] == ["buy", "hold", "sell"]

    agent = only_span(exporter, "decision_agent")
    assert agent.attributes["openinference.span.kind"] == "AGENT"
    assert agent.attributes["finagent.decision"] == "buy"
    assert agent.attributes["finagent.confidence"] == 0.8
    assert agent.attributes["finagent.agent.iterations"] == 2
    assert [e.name for e in agent.events] == ["decision.submitted"]
    assert len(spans_named(exporter, "anthropic.messages")) == 2
    fi = only_span(exporter, "force_index")
    assert fi.parent.span_id == agent.context.span_id
    assert fi.attributes["finagent.fake.force_index"] == "ran"
    llm = spans_named(exporter, "anthropic.messages")[0]
    assert llm.attributes["llm.prompt_template.version"] == "v1"
    assert "{ticker}" in llm.attributes["llm.prompt_template.template"]


async def test_failing_tool_is_reported_to_model_and_span(exporter):
    tools = {**TOOLS, "insider_trades": fake_local_tool("insider_trades", {}, error=ValueError("no data"))}
    client = FakeAnthropic(
        [
            message(tool_use("tu_1", "insider_trades", {"ticker": "SPY"}), stop_reason="tool_use"),
            message(submit("hold", 0.5), stop_reason="tool_use"),
        ]
    )
    result = await run_decision_agent("SPY", "brief", client=client, model="m", tools=tools)
    assert result["decision"] == "hold"
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "no data" in tool_result["content"]
    assert only_span(exporter, "insider_trades").status.status_code is StatusCode.ERROR


async def test_unknown_tool_is_reported_to_model(exporter):
    client = FakeAnthropic(
        [
            message(tool_use("tu_1", "rsi", {"ticker": "AAPL"}), stop_reason="tool_use"),
            message(submit(), stop_reason="tool_use"),
        ]
    )
    await run_decision_agent("AAPL", "brief", client=client, model="m", tools=TOOLS)
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


async def test_nudges_when_model_stops_without_decision(exporter):
    client = FakeAnthropic([message(text("I think buy.")), message(submit(), stop_reason="tool_use")])
    result = await run_decision_agent("AAPL", "brief", client=client, model="m", tools=TOOLS)
    assert result["decision"] == "buy"
    assert client.calls[1]["messages"][-1] == {
        "role": "user",
        "content": "Call submit_decision now with your final decision.",
    }


async def test_rejects_invalid_decision_then_accepts_valid(exporter):
    client = FakeAnthropic(
        [
            message(submit("strong buy", id="tu_bad"), stop_reason="tool_use"),
            message(submit("sell", 1.7), stop_reason="tool_use"),
        ]
    )
    result = await run_decision_agent("AAPL", "brief", client=client, model="m", tools=TOOLS)
    assert result["decision"] == "sell"
    assert result["confidence"] == 1.0  # clamped
    rejected = client.calls[1]["messages"][-1]["content"][0]
    assert rejected["is_error"] is True
    statuses = [s.status.status_code for s in spans_named(exporter, "submit_decision")]
    assert statuses == [StatusCode.ERROR, StatusCode.OK]


async def test_gives_up_after_max_iterations(exporter):
    client = FakeAnthropic([message(text("hmm")) for _ in range(MAX_ITERATIONS)])
    with pytest.raises(DecisionAgentError):
        await run_decision_agent("AAPL", "brief", client=client, model="m", tools=TOOLS)
    agent = only_span(exporter, "decision_agent")
    assert agent.status.status_code is StatusCode.ERROR
    assert "agent.max_iterations_reached" in [e.name for e in agent.events]


async def test_refusal_fails_the_agent(exporter):
    client = FakeAnthropic([message(text(""), stop_reason="refusal")])
    with pytest.raises(DecisionAgentError, match="refused"):
        await run_decision_agent("AAPL", "brief", client=client, model="m", tools=TOOLS)


async def test_app_continues_caller_trace_from_headers(exporter):
    client = FakeAnthropic([message(submit(), stop_reason="tool_use")])
    app = create_app(client=client, model="m", tools=TOOLS)
    with request_context(session_id="sess-9", user_id="u", request_id="r", ticker="AAPL"):
        with chain_span("caller", input="x"):
            headers = inject_carrier()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://decision") as http:
        response = await http.post("/invoke", json={"ticker": "AAPL", "research_brief": "b"}, headers=headers)
    assert response.status_code == 200
    assert response.json()["decision"] == "buy"
    agent, caller = only_span(exporter, "decision_agent"), only_span(exporter, "caller")
    assert agent.parent.span_id == caller.context.span_id
    assert agent.attributes["session.id"] == "sess-9"


async def test_app_rejects_bad_ticker_and_reports_agent_failure(exporter):
    client = FakeAnthropic([message(text("")) for _ in range(MAX_ITERATIONS)])
    app = create_app(client=client, model="m", tools=TOOLS)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://decision") as http:
        bad = await http.post("/invoke", json={"ticker": "bad ticker!", "research_brief": "b"})
        failed = await http.post("/invoke", json={"ticker": "AAPL", "research_brief": "b"})
        health = await http.get("/health")
    assert bad.status_code == 422
    assert failed.status_code == 502
    assert health.json() == {"status": "ok"}
