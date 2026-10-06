"""Carry trace context *and* OpenInference request attributes across process boundaries.

traceparent alone links spans, but using_attributes() lives only in in-process context.
So every hop also sends W3C baggage, and the receiver rebuilds the same context attributes.
"""

import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from openinference.instrumentation import using_attributes
from opentelemetry import baggage, propagate
from opentelemetry import context as otel_context

from fin_tracing.spans import TICKER_KEY

SESSION_KEY = "finagent.session_id"
USER_KEY = "finagent.user_id"
REQUEST_KEY = "finagent.request_id"
TAGS = ["financial-analysis", "multi-agent", "demo"]
APP_VERSION = "0.1.0"


@contextmanager
def request_context(*, session_id: str, user_id: str, request_id: str, ticker: str) -> Iterator[None]:
    ctx = otel_context.get_current()
    for key, value in (
        (SESSION_KEY, session_id),
        (USER_KEY, user_id),
        (REQUEST_KEY, request_id),
        (TICKER_KEY, ticker),
    ):
        ctx = baggage.set_baggage(key, value, context=ctx)
    token = otel_context.attach(ctx)
    try:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata={"request_id": request_id, "ticker": ticker, "app_version": APP_VERSION},
            tags=TAGS,
        ):
            yield
    finally:
        otel_context.detach(token)


def inject_carrier() -> dict[str, str]:
    """traceparent + baggage for the current context, for HTTP headers or MCP _meta."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return carrier


@contextmanager
def continue_request_context(carrier: Mapping[str, Any], *, ticker: str | None = None) -> Iterator[None]:
    """Continue the caller's trace and request attributes from HTTP headers or MCP _meta."""
    text_carrier = {k.lower(): v for k, v in carrier.items() if isinstance(v, str)}
    token = otel_context.attach(propagate.extract(text_carrier))
    try:
        with request_context(
            session_id=str(baggage.get_baggage(SESSION_KEY) or uuid.uuid4()),
            user_id=str(baggage.get_baggage(USER_KEY) or "anonymous"),
            request_id=str(baggage.get_baggage(REQUEST_KEY) or uuid.uuid4()),
            ticker=str(baggage.get_baggage(TICKER_KEY) or ticker or "UNKNOWN"),
        ):
            yield
    finally:
        otel_context.detach(token)
