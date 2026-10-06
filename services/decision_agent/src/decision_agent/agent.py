"""Decision agent: a Claude tool-use loop over local indicator tools, ending in submit_decision."""

import asyncio
import json
from typing import Any

from openinference.instrumentation import using_prompt_template

from decision_agent.tools import LOCAL_TOOLS, LocalTool
from fin_tracing import agent_span, tool_span, traced_messages_create

DECISIONS = ("buy", "hold", "sell")
MAX_ITERATIONS = 8
PROMPT_VERSION = "v1"
NUDGE = "Call submit_decision now with your final decision."

SYSTEM_PROMPT_TEMPLATE = (
    "You are a disciplined equity trading decision agent. Decide whether to buy, hold or sell {ticker}.\n"
    "You receive a research brief written by a research analyst. Before deciding, call the force_index, "
    "bollinger_bands and insider_trades tools for {ticker}, then weigh those signals together with the brief.\n"
    "Finish by calling submit_decision exactly once with your decision, a confidence between 0 and 1, "
    "and a short rationale that cites the indicator values you relied on."
)

SUBMIT_DECISION_DESCRIPTION = (
    "Submit the final trading decision. Call this exactly once, after using the analysis tools."
)
SUBMIT_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(DECISIONS)},
        "confidence": {"type": "number", "description": "Between 0 and 1"},
        "rationale": {"type": "string"},
    },
    "required": ["decision", "confidence", "rationale"],
    "additionalProperties": False,
}
# Forced tool_choice is rejected by current models, so a strict schema + the prompt steer the call.
SUBMIT_DECISION_TOOL: dict[str, Any] = {
    "name": "submit_decision",
    "description": SUBMIT_DECISION_DESCRIPTION,
    "input_schema": SUBMIT_DECISION_SCHEMA,
    "strict": True,
}


class DecisionAgentError(RuntimeError):
    pass


def _tool_result(tool_use_id: str, content: str, *, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


async def _run_local_tool(tool: LocalTool | None, block: Any, indicators: dict[str, Any]) -> dict[str, Any]:
    if tool is None:
        return _tool_result(block.id, f"Unknown tool: {block.name}", is_error=True)
    try:
        with tool_span(
            tool.name,
            description=tool.description,
            parameters_schema=tool.input_schema,
            arguments=block.input,
        ) as span:
            # yfinance is blocking; to_thread copies contextvars so span events still attach.
            result = await asyncio.to_thread(tool.run, **block.input)
            span.set_attributes(tool.span_attributes(result))
            span.set_output(result)
    except Exception as exc:
        return _tool_result(block.id, f"{type(exc).__name__}: {exc}", is_error=True)
    indicators[tool.name] = result
    return _tool_result(block.id, json.dumps(result, default=str))


def _submit_decision(block: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Validate a submit_decision call. Returns (decision or None, tool_result block)."""
    with tool_span(
        "submit_decision",
        description=SUBMIT_DECISION_DESCRIPTION,
        parameters_schema=SUBMIT_DECISION_SCHEMA,
        arguments=block.input,
    ) as span:
        decision = str(block.input.get("decision", "")).strip().lower()
        if decision not in DECISIONS:
            span.fail(f"invalid decision {decision!r}")
            return None, _tool_result(block.id, f"decision must be one of {list(DECISIONS)}", is_error=True)
        try:
            confidence = min(max(float(block.input.get("confidence", 0.5)), 0.0), 1.0)
        except (TypeError, ValueError):
            confidence = 0.5
        result = {"decision": decision, "confidence": confidence, "rationale": str(block.input.get("rationale", ""))}
        span.set_output(result)
        return result, _tool_result(block.id, "Decision recorded.")


async def run_decision_agent(
    ticker: str,
    research_brief: str,
    *,
    client: Any,
    model: str,
    tools: dict[str, LocalTool] = LOCAL_TOOLS,
) -> dict[str, Any]:
    variables = {"ticker": ticker}
    system = SYSTEM_PROMPT_TEMPLATE.format(**variables)
    tool_definitions = [t.definition() for t in tools.values()] + [SUBMIT_DECISION_TOOL]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Ticker: {ticker}\n\nResearch brief:\n{research_brief}"}
    ]
    indicators: dict[str, Any] = {}

    with agent_span("decision_agent", input={"ticker": ticker, "research_brief": research_brief}) as span:
        for iteration in range(1, MAX_ITERATIONS + 1):
            with using_prompt_template(template=SYSTEM_PROMPT_TEMPLATE, variables=variables, version=PROMPT_VERSION):
                response = await traced_messages_create(
                    client, model=model, system=system, messages=messages, tools=tool_definitions
                )
            if response.stop_reason == "refusal":
                raise DecisionAgentError("Model refused to produce a decision")
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                messages.append({"role": "user", "content": NUDGE})
                continue

            results = []
            for block in tool_uses:
                if block.name != SUBMIT_DECISION_TOOL["name"]:
                    results.append(await _run_local_tool(tools.get(block.name), block, indicators))
                    continue
                decision, result_block = _submit_decision(block)
                if decision is None:
                    results.append(result_block)
                    continue
                decision["indicators"] = indicators
                span.set_attributes(
                    {
                        "finagent.decision": decision["decision"],
                        "finagent.confidence": decision["confidence"],
                        "finagent.agent.iterations": iteration,
                    }
                )
                span.add_event(
                    "decision.submitted",
                    {"decision": decision["decision"], "confidence": decision["confidence"]},
                )
                span.set_output(decision)
                return decision
            messages.append({"role": "user", "content": results})

        span.add_event("agent.max_iterations_reached", {"max_iterations": MAX_ITERATIONS})
        raise DecisionAgentError(f"No decision after {MAX_ITERATIONS} iterations")
