"""Local tools for the decision agent: Anthropic tool definitions plus implementations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from decision_agent import market_data
from decision_agent.indicators import bollinger_bands, force_index, summarize_insider_transactions

TICKER_PROPERTY = {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}


@dataclass(frozen=True)
class LocalTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[..., dict[str, Any]]
    # Maps the tool result to vendored finagent.* span attributes.
    span_attributes: Callable[[dict[str, Any]], dict[str, Any]]

    def definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


def run_force_index(ticker: str, period: int = 13) -> dict[str, Any]:
    history = market_data.fetch_price_history(ticker)
    return {"ticker": ticker, **force_index(history["Close"], history["Volume"], period)}


def run_bollinger_bands(ticker: str, window: int = 20, num_std: float = 2.0) -> dict[str, Any]:
    history = market_data.fetch_price_history(ticker)
    return {"ticker": ticker, **bollinger_bands(history["Close"], window, num_std)}


def run_insider_trades(ticker: str, days: int = 90) -> dict[str, Any]:
    transactions = market_data.fetch_insider_transactions(ticker)
    summary = summarize_insider_transactions(transactions, days=days, today=pd.Timestamp.today())
    return {"ticker": ticker, **summary}


FORCE_INDEX = LocalTool(
    name="force_index",
    description=(
        "Elder's Force Index (price change x volume) over ~6 months of daily data, smoothed with an EMA. "
        "Positive values mean buying pressure, negative values mean selling pressure."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ticker": TICKER_PROPERTY,
            "period": {"type": "integer", "minimum": 1, "maximum": 50, "default": 13},
        },
        "required": ["ticker"],
    },
    run=run_force_index,
    span_attributes=lambda r: {
        "finagent.force_index.ema": r["force_index_ema"],
        "finagent.force_index.trend": r["trend"],
    },
)

BOLLINGER_BANDS = LocalTool(
    name="bollinger_bands",
    description=(
        "Bollinger Bands on daily closes: upper/middle/lower band, %B and bandwidth. "
        "%B >= 1 is overbought, %B <= 0 is oversold."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ticker": TICKER_PROPERTY,
            "window": {"type": "integer", "minimum": 2, "maximum": 100, "default": 20},
            "num_std": {"type": "number", "minimum": 0.5, "maximum": 4, "default": 2.0},
        },
        "required": ["ticker"],
    },
    run=run_bollinger_bands,
    span_attributes=lambda r: {
        "finagent.bollinger.percent_b": r["percent_b"],
        "finagent.bollinger.signal": r["signal"],
    },
)

INSIDER_TRADES = LocalTool(
    name="insider_trades",
    description=(
        "Insider purchases and sales over the last N days: counts, net shares, dollar values "
        "and the largest transactions."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ticker": TICKER_PROPERTY,
            "days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 90},
        },
        "required": ["ticker"],
    },
    run=run_insider_trades,
    span_attributes=lambda r: {
        "finagent.insider.net_shares": r["net_shares"],
        "finagent.insider.txn_count": r["buy_count"] + r["sell_count"] + r["other_count"],
    },
)

LOCAL_TOOLS: dict[str, LocalTool] = {t.name: t for t in (FORCE_INDEX, BOLLINGER_BANDS, INSIDER_TRADES)}
