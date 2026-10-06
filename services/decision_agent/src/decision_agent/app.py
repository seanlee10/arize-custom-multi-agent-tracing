"""FastAPI entry point for the decision-agent container."""

import os
from contextlib import asynccontextmanager
from typing import Any, Literal

import anthropic
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from decision_agent.agent import DecisionAgentError, run_decision_agent
from decision_agent.tools import LOCAL_TOOLS, LocalTool
from fin_tracing import DEFAULT_MODEL, continue_request_context, setup_tracing


class DecideRequest(BaseModel):
    ticker: str = Field(pattern=r"^[A-Z.\-]{1,10}$")
    research_brief: str


class DecideResponse(BaseModel):
    decision: Literal["buy", "hold", "sell"]
    confidence: float
    rationale: str
    indicators: dict[str, Any]


def create_app(*, client: Any, model: str, tools: dict[str, LocalTool] = LOCAL_TOOLS, lifespan=None) -> FastAPI:
    app = FastAPI(title="decision-agent", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/invoke", response_model=DecideResponse)
    async def invoke(body: DecideRequest, request: Request) -> dict[str, Any]:
        # Headers carry traceparent + baggage from the research agent.
        with continue_request_context(request.headers, ticker=body.ticker):
            try:
                return await run_decision_agent(
                    body.ticker, body.research_brief, client=client, model=model, tools=tools
                )
            except (DecisionAgentError, anthropic.APIError) as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


def create_app_from_env() -> FastAPI:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    provider = setup_tracing("decision-agent")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        provider.shutdown()

    return create_app(
        client=anthropic.AsyncAnthropic(),
        model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
        lifespan=lifespan,
    )
