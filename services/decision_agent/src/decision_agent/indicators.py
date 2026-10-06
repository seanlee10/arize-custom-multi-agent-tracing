"""Pure indicator math (no I/O) so it can be unit tested with fixed series."""

from typing import Any

import pandas as pd


def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> dict[str, Any]:
    """Elder's Force Index: (close_t - close_t-1) * volume_t, smoothed with EMAs."""
    if period < 1:
        raise ValueError("period must be >= 1")
    if len(close) < period + 1:
        raise ValueError(f"need at least {period + 1} closes, got {len(close)}")
    raw = (close.diff() * volume).dropna()
    ema_short = float(raw.ewm(span=2, adjust=False).mean().iloc[-1])
    ema = float(raw.ewm(span=period, adjust=False).mean().iloc[-1])
    trend = "bullish" if ema > 0 else "bearish" if ema < 0 else "neutral"
    return {
        "period": period,
        "force_index_raw": float(raw.iloc[-1]),
        "force_index_ema2": ema_short,
        "force_index_ema": ema,
        "trend": trend,
    }


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> dict[str, Any]:
    """Bollinger Bands over the last `window` closes (population std, per Bollinger)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    if len(close) < window:
        raise ValueError(f"need at least {window} closes, got {len(close)}")
    recent = close.iloc[-window:]
    middle = float(recent.mean())
    std = float(recent.std(ddof=0))
    upper, lower = middle + num_std * std, middle - num_std * std
    last = float(close.iloc[-1])
    percent_b = 0.5 if upper == lower else (last - lower) / (upper - lower)
    signal = "overbought" if percent_b >= 1 else "oversold" if percent_b <= 0 else "neutral"
    return {
        "window": window,
        "num_std": num_std,
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "last_close": last,
        "percent_b": percent_b,
        "bandwidth": (upper - lower) / middle if middle else 0.0,
        "signal": signal,
    }


def summarize_insider_transactions(
    transactions: pd.DataFrame | None, *, days: int, today: pd.Timestamp
) -> dict[str, Any]:
    """Aggregate yfinance `insider_transactions` rows within the last `days` days."""
    summary: dict[str, Any] = {
        "window_days": days,
        "buy_count": 0,
        "sell_count": 0,
        "other_count": 0,
        "net_shares": 0,
        "buy_value": 0.0,
        "sell_value": 0.0,
        "top_transactions": [],
    }
    if transactions is None or transactions.empty:
        return summary
    df = transactions.fillna({"Shares": 0, "Value": 0.0, "Text": "", "Insider": "", "Position": ""})
    df = df.assign(**{"Start Date": pd.to_datetime(df["Start Date"])})
    df = df[df["Start Date"] >= today.normalize() - pd.Timedelta(days=days)]
    if df.empty:
        return summary
    text = df["Text"].str.lower()
    buys, sells = df[text.str.startswith("purchase")], df[text.str.startswith("sale")]
    top = df.sort_values("Value", ascending=False).head(5)
    summary.update(
        buy_count=len(buys),
        sell_count=len(sells),
        other_count=len(df) - len(buys) - len(sells),
        net_shares=int(buys["Shares"].sum() - sells["Shares"].sum()),
        buy_value=float(buys["Value"].sum()),
        sell_value=float(sells["Value"].sum()),
        top_transactions=[
            {
                "date": row["Start Date"].date().isoformat(),
                "insider": row["Insider"],
                "position": row["Position"],
                "text": row["Text"],
                "shares": int(row["Shares"]),
                "value": float(row["Value"]),
            }
            for _, row in top.iterrows()
        ],
    )
    return summary
