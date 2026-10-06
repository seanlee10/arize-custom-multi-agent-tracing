"""Research agent: gathers information with the MCP web_search tool, then asks the decision agent."""

import json
from dataclasses import dataclass
from typing import Any

import httpx
from mcp.server.mcpserver import MCPServer
from openinference.instrumentation import using_prompt_template
from opentelemetry import trace

from fin_tracing import agent_span, traced_messages_create
from research_agent.decision_client import invoke_decision_agent
from research_agent.mcp_client import WebSearchClient

MAX_SEARCHES = 4
MAX_ITERATIONS = 8
PROMPT_VERSION = "v1"

SYSTEM_PROMPT_TEMPLATE = (
    "You are a financial research analyst. Research the stock {ticker} using the web_search tool "
    "(at most {max_searches} searches). Look for recent news, the latest earnings and guidance, "
    "analyst sentiment and price targets, and key risks.\n"
    "Then write a research brief of at most 300 words with the sections Summary, Recent News, "
    "Fundamentals, Sentiment and Risks. Do not make a buy, hold or sell recommendation; "
    "a separate decision agent does that."
)


@dataclass
class ResearchDeps:
    client: Any
    model: str
    websearch_server: str | MCPServer
    decision_url: str
    decision_transport: httpx.AsyncBaseTransport | None = None


class AnalysisError(RuntimeError):
    def __init__(self, message: str, trace_id: str) -> None:
        super().__init__(message)
        self.trace_id = trace_id


async def _call_mcp_tool(web_search: WebSearchClient, block: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.id}
    try:
        output = await web_search.call(block.name, block.input)
    except Exception as exc:
        return {**result, "content": f"{type(exc).__name__}: {exc}", "is_error": True}
    return {**result, "content": json.dumps(output)}


async def gather_research(ticker: str, *, client: Any, model: str, web_search: WebSearchClient) -> str:
    variables = {"ticker": ticker, "max_searches": MAX_SEARCHES}
    system = SYSTEM_PROMPT_TEMPLATE.format(**variables)
    tools = web_search.tool_definitions()
    messages: list[dict[str, Any]] = [{"role": "user", "content": f"Write the research brief for {ticker}."}]
    agent = trace.get_current_span()

    for iteration in range(1, MAX_ITERATIONS + 1):
        with using_prompt_template(template=SYSTEM_PROMPT_TEMPLATE, variables=variables, version=PROMPT_VERSION):
            response = await traced_messages_create(
                client, model=model, system=system, messages=messages, tools=tools
            )
        if response.stop_reason == "refusal":
            raise RuntimeError("Model refused the research request")
        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            agent.set_attribute("finagent.agent.iterations", iteration)
            return "\n".join(b.text for b in response.content if b.type == "text").strip()
        messages.append({"role": "user", "content": [await _call_mcp_tool(web_search, b) for b in tool_uses]})

    agent.add_event("agent.max_iterations_reached", {"max_iterations": MAX_ITERATIONS})
    raise RuntimeError(f"No research brief after {MAX_ITERATIONS} iterations")


async def analyze_ticker(ticker: str, deps: ResearchDeps) -> dict[str, Any]:
    with agent_span("research_agent", input={"ticker": ticker}) as span:
        try:
            async with WebSearchClient(deps.websearch_server) as web_search:
                brief = await gather_research(ticker, client=deps.client, model=deps.model, web_search=web_search)
            decision = await invoke_decision_agent(
                deps.decision_url, ticker=ticker, research_brief=brief, transport=deps.decision_transport
            )
        except Exception as exc:
            raise AnalysisError(f"{type(exc).__name__}: {exc}", span.trace_id) from exc
        result = {
            "ticker": ticker,
            "decision": decision["decision"],
            "confidence": decision["confidence"],
            "rationale": decision["rationale"],
            "research_brief": brief,
            "trace_id": span.trace_id,
        }
        span.set_attributes({"finagent.decision": result["decision"], "finagent.confidence": result["confidence"]})
        span.set_output(result)
        return result
