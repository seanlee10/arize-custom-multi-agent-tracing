"""yfinance I/O. Emits span events on the current (TOOL) span."""

import pandas as pd
import yfinance as yf
from opentelemetry import trace


def fetch_price_history(ticker: str, period: str = "6mo") -> pd.DataFrame:
    history = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if history.empty:
        raise ValueError(f"No price history found for {ticker}")
    trace.get_current_span().add_event(
        "price_history.fetched",
        {
            "rows": len(history),
            "start": history.index[0].date().isoformat(),
            "end": history.index[-1].date().isoformat(),
        },
    )
    return history


def fetch_insider_transactions(ticker: str) -> pd.DataFrame | None:
    transactions = yf.Ticker(ticker).insider_transactions
    rows = 0 if transactions is None else len(transactions)
    trace.get_current_span().add_event("insider_data.fetched", {"rows": rows})
    return transactions
