"""FastAPI entry point for the research-agent container (the system's front door)."""

import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from fin_tracing import DEFAULT_MODEL, request_context, setup_tracing
from research_agent.agent import AnalysisError, ResearchDeps, analyze_ticker

TICKER_PATTERN = re.compile(r"^[A-Z.\-]{1,10}$")


class InvokeRequest(BaseModel):
    ticker: str
    session_id: str | None = None
    user_id: str | None = None

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, value: str) -> str:
        value = value.strip().upper()
        if not TICKER_PATTERN.match(value):
            raise ValueError("ticker must be 1-10 characters of A-Z, '.' or '-'")
        return value


class InvokeResponse(BaseModel):
    ticker: str
    decision: Literal["buy", "hold", "sell"]
    confidence: float
    rationale: str
    research_brief: str
    trace_id: str


def create_app(deps: ResearchDeps, *, lifespan=None) -> FastAPI:
    app = FastAPI(title="research-agent", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(body: InvokeRequest) -> dict[str, Any]:
        with request_context(
            session_id=body.session_id or str(uuid.uuid4()),
            user_id=body.user_id or "demo-user",
            request_id=str(uuid.uuid4()),
            ticker=body.ticker,
        ):
            try:
                return await analyze_ticker(body.ticker, deps)
            except AnalysisError as exc:
                raise HTTPException(status_code=502, detail={"error": str(exc), "trace_id": exc.trace_id}) from exc

    return app


def create_app_from_env() -> FastAPI:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is required")
    provider = setup_tracing("research-agent")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        provider.shutdown()

    deps = ResearchDeps(
        client=anthropic.AsyncAnthropic(),
        model=os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL),
        websearch_server=os.getenv("WEBSEARCH_MCP_URL", "http://localhost:8000/mcp"),
        decision_url=os.getenv("DECISION_AGENT_URL", "http://localhost:8002"),
    )
    return create_app(deps, lifespan=lifespan)
