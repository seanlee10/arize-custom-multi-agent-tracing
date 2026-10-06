import json

from openinference.semconv.trace import (
    MessageAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)
from opentelemetry.trace import StatusCode

from fin_tracing.anthropic_attrs import (
    request_attributes,
    response_attributes,
    to_openinference_messages,
)
from fin_tracing.llm import FALLBACK_BETA, traced_messages_create
from tests.fakes import FakeAnthropic, message, text, tool_use
from tests.helpers import only_span

IN = SpanAttributes.LLM_INPUT_MESSAGES
OUT = SpanAttributes.LLM_OUTPUT_MESSAGES
ROLE = MessageAttributes.MESSAGE_ROLE
CONTENT = MessageAttributes.MESSAGE_CONTENT
CALLS = MessageAttributes.MESSAGE_TOOL_CALLS

CONVERSATION = [
    {"role": "user", "content": "Research AAPL"},
    {
        "role": "assistant",
        "content": message(text("Searching."), tool_use("tu_1", "web_search", {"query": "AAPL"})).content,
    },
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "[]"}]},
]


def test_to_openinference_messages_flattens_tool_calls_and_results():
    assert to_openinference_messages("Be terse.", CONVERSATION) == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Research AAPL"},
        {
            "role": "assistant",
            "content": "Searching.",
            "tool_calls": [{"id": "tu_1", "name": "web_search", "arguments": '{"query": "AAPL"}'}],
        },
        {"role": "tool", "tool_call_id": "tu_1", "content": "[]"},
    ]


def test_thinking_blocks_are_skipped():
    content = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "hi"}]
    assert to_openinference_messages(None, [{"role": "assistant", "content": content}]) == [
        {"role": "assistant", "content": "hi"}
    ]


def test_request_attributes():
    tools = [{"name": "web_search", "description": "d", "input_schema": {"type": "object"}}]
    attrs = request_attributes(
        model="claude-opus-5-5",
        system="Be terse.",
        messages=CONVERSATION,
        tools=tools,
        invocation_parameters={"max_tokens": 100},
    )
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "claude-opus-5-5"
    assert attrs[SpanAttributes.LLM_PROVIDER] == "anthropic"
    assert attrs[SpanAttributes.LLM_SYSTEM] == "anthropic"
    assert json.loads(attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS]) == {"max_tokens": 100}
    assert attrs[f"{IN}.0.{ROLE}"] == "system"
    assert attrs[f"{IN}.2.{CONTENT}"] == "Searching."
    assert attrs[f"{IN}.2.{CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "web_search"
    assert attrs[f"{IN}.2.{CALLS}.0.{ToolCallAttributes.TOOL_CALL_ID}"] == "tu_1"
    assert attrs[f"{IN}.3.{ROLE}"] == "tool"
    assert attrs[f"{IN}.3.{MessageAttributes.MESSAGE_TOOL_CALL_ID}"] == "tu_1"
    assert json.loads(attrs[f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"]) == tools[0]
    assert attrs[SpanAttributes.INPUT_MIME_TYPE] == "application/json"


def test_response_attributes_counts_cached_prompt_tokens():
    response = message(text("Brief."), input_tokens=100, output_tokens=20, cache_read=50)
    attrs = response_attributes(response)
    assert attrs[f"{OUT}.0.{ROLE}"] == "assistant"
    assert attrs[f"{OUT}.0.{CONTENT}"] == "Brief."
    assert attrs[SpanAttributes.OUTPUT_VALUE] == "Brief."
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 150
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 170
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 50
    assert attrs["finagent.llm.stop_reason"] == "end_turn"


async def test_traced_messages_create_emits_llm_span(exporter):
    client = FakeAnthropic([message(text("ok"))])
    response = await traced_messages_create(
        client, model="claude-opus-5-5", system="sys", messages=[{"role": "user", "content": "hi"}], tools=[]
    )
    assert response.content[0].text == "ok"
    call = client.calls[0]
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert call["output_config"] == {"effort": "medium"}
    assert "tool_choice" not in call and "thinking" not in call
    span = only_span(exporter, "anthropic.messages")
    assert span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "LLM"
    assert span.attributes[f"{IN}.1.{CONTENT}"] == "hi"
    assert span.attributes[f"{OUT}.0.{CONTENT}"] == "ok"
    assert span.status.status_code is StatusCode.OK


async def test_refusal_marks_llm_span_error(exporter):
    client = FakeAnthropic([message(text(""), stop_reason="refusal")])
    await traced_messages_create(client, model="m", system="s", messages=[{"role": "user", "content": "x"}], tools=[])
    assert only_span(exporter, "anthropic.messages").status.status_code is StatusCode.ERROR
