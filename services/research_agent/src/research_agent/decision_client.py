"""HTTP call from the research agent to the decision agent, wrapped in a CHAIN span."""

from typing import Any

import httpx

from fin_tracing import chain_span, inject_carrier


async def invoke_decision_agent(
    base_url: str,
    *,
    ticker: str,
    research_brief: str,
    timeout_seconds: float = 120.0,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    payload = {"ticker": ticker, "research_brief": research_brief}
    with chain_span("invoke_decision_agent", input=payload) as span:
        span.set_attributes({"finagent.decision_agent.url": base_url})
        async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, transport=transport) as http:
            response = await http.post("/invoke", json=payload, headers=inject_carrier())
        span.set_attributes({"http.response.status_code": response.status_code})
        response.raise_for_status()
        result = response.json()
        span.set_output(result)
        return result
