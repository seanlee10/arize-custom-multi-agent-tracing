"""Map Anthropic Messages API requests/responses onto OpenInference LLM span attributes."""

import json
from collections.abc import Iterable, Mapping
from typing import Any

from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


def _as_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, Mapping):
        return dict(block)
    return block.model_dump(exclude_none=True)


def _tool_result_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(_as_dict(block).get("text", "") for block in content)


def to_openinference_messages(
    system: str | None, messages: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Flatten Anthropic messages into OpenInference messages.

    One Anthropic message can become several: each tool_result block becomes its own
    role="tool" message so Arize can pair it with the tool call it answers.
    """
    result: list[dict[str, Any]] = []
    if system:
        result.append({"role": "system", "content": system})
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue
        texts: list[str] = []
        tool_calls: list[dict[str, str]] = []
        for block in map(_as_dict, content):
            kind = block.get("type")
            if kind == "text":
                texts.append(block["text"])
            elif kind == "tool_use":
                tool_calls.append(
                    {"id": block["id"], "name": block["name"], "arguments": json.dumps(block.get("input", {}))}
                )
            elif kind == "tool_result":
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": _tool_result_text(block.get("content")),
                    }
                )
            # thinking / fallback blocks carry no user-visible text; skip them
        if texts or tool_calls:
            entry: dict[str, Any] = {"role": role, "content": "\n".join(texts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            result.append(entry)
    return result


def _message_attributes(prefix: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for i, message in enumerate(messages):
        base = f"{prefix}.{i}."
        attrs[base + MessageAttributes.MESSAGE_ROLE] = message["role"]
        if message.get("content"):
            attrs[base + MessageAttributes.MESSAGE_CONTENT] = message["content"]
        if "tool_call_id" in message:
            attrs[base + MessageAttributes.MESSAGE_TOOL_CALL_ID] = message["tool_call_id"]
        for j, call in enumerate(message.get("tool_calls", [])):
            call_base = f"{base}{MessageAttributes.MESSAGE_TOOL_CALLS}.{j}."
            attrs[call_base + ToolCallAttributes.TOOL_CALL_ID] = call["id"]
            attrs[call_base + ToolCallAttributes.TOOL_CALL_FUNCTION_NAME] = call["name"]
            attrs[call_base + ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON] = call["arguments"]
    return attrs


def request_attributes(
    *,
    model: str,
    system: str | None,
    messages: Iterable[Mapping[str, Any]],
    tools: list[dict[str, Any]],
    invocation_parameters: dict[str, Any],
) -> dict[str, Any]:
    oi_messages = to_openinference_messages(system, messages)
    attrs: dict[str, Any] = {
        SpanAttributes.LLM_MODEL_NAME: model,
        SpanAttributes.LLM_PROVIDER: OpenInferenceLLMProviderValues.ANTHROPIC.value,
        SpanAttributes.LLM_SYSTEM: OpenInferenceLLMSystemValues.ANTHROPIC.value,
        SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(invocation_parameters),
        SpanAttributes.INPUT_VALUE: json.dumps(oi_messages),
        SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
        **_message_attributes(SpanAttributes.LLM_INPUT_MESSAGES, oi_messages),
    }
    for k, tool in enumerate(tools):
        attrs[f"{SpanAttributes.LLM_TOOLS}.{k}.{ToolAttributes.TOOL_JSON_SCHEMA}"] = json.dumps(tool)
    return attrs


def response_attributes(response: Any) -> dict[str, Any]:
    oi_messages = to_openinference_messages(None, [{"role": "assistant", "content": response.content}])
    blocks = [_as_dict(b) for b in response.content]
    output_text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
    usage = response.usage
    cache_read = getattr(usage, "cache_read_input_tokens", None) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", None) or 0
    prompt_tokens = usage.input_tokens + cache_read + cache_write
    attrs: dict[str, Any] = {
        # response.model reflects the model that actually served the turn (fallbacks may switch it)
        SpanAttributes.LLM_MODEL_NAME: response.model,
        SpanAttributes.OUTPUT_VALUE: output_text or json.dumps(oi_messages),
        SpanAttributes.OUTPUT_MIME_TYPE: (
            OpenInferenceMimeTypeValues.TEXT.value if output_text else OpenInferenceMimeTypeValues.JSON.value
        ),
        **_message_attributes(SpanAttributes.LLM_OUTPUT_MESSAGES, oi_messages),
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: prompt_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: usage.output_tokens,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: prompt_tokens + usage.output_tokens,
        "finagent.llm.stop_reason": response.stop_reason or "",
    }
    if cache_read:
        attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] = cache_read
    return attrs
