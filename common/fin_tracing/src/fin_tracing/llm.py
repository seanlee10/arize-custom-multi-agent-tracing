"""An LLM span around one Claude Messages API call."""

from typing import Any

from fin_tracing.anthropic_attrs import request_attributes, response_attributes
from fin_tracing.spans import llm_span

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


async def traced_messages_create(
    client: Any,
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int = 16000,
    effort: str = "medium",
) -> Any:
    """Call client.beta.messages.create (an AsyncAnthropic) inside an OpenInference LLM span."""
    output_config = {"effort": effort}
    with llm_span() as span:
        span.set_attributes(
            request_attributes(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                invocation_parameters={
                    "max_tokens": max_tokens,
                    "output_config": output_config,
                    "fallbacks": "default",
                },
            )
        )
        response = await client.beta.messages.create(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            output_config=output_config,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        span.set_attributes(response_attributes(response))
        if response.stop_reason == "refusal":
            span.fail("model refused the request")
        return response
