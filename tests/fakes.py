from types import SimpleNamespace
from typing import Any

from anthropic.types.beta import BetaMessage


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def tool_use(id: str, name: str, input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "tool_use", "id": id, "name": name, "input": input}


def message(
    *blocks: dict[str, Any],
    stop_reason: str = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
    cache_read: int | None = None,
) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
            },
        }
    )


class FakeAnthropic:
    """Stands in for AsyncAnthropic: returns scripted responses from beta.messages.create."""

    def __init__(self, responses: list[BetaMessage]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> BetaMessage:
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self._responses:
            raise AssertionError("FakeAnthropic ran out of scripted responses")
        return self._responses.pop(0)


def fake_local_tool(name: str, result: dict[str, Any], *, error: Exception | None = None):
    from decision_agent.tools import LocalTool

    def run(**kwargs: Any) -> dict[str, Any]:
        if error is not None:
            raise error
        return {**result, "args": kwargs}

    return LocalTool(
        name=name,
        description=f"fake {name}",
        input_schema={"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
        run=run,
        span_attributes=lambda r: {f"finagent.fake.{name}": "ran"},
    )
