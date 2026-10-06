from statistics import pstdev

import pandas as pd
import pytest

from decision_agent.indicators import bollinger_bands, force_index, summarize_insider_transactions


def series(*values: float) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_force_index_neutral_values():
    # raw FI = [100, 100, -100]; EMA(3) alpha=0.5 -> 100, 100, 0; EMA(2) alpha=2/3 -> -33.33
    result = force_index(series(10, 11, 12, 11), series(100, 100, 100, 100), period=3)
    assert result["force_index_raw"] == -100
    assert result["force_index_ema"] == 0
    assert result["force_index_ema2"] == pytest.approx(-33.3333, rel=1e-4)
    assert result["trend"] == "neutral"


def test_force_index_trend_direction():
    volume = series(100, 100, 100, 100)
    assert force_index(series(10, 11, 12, 13), volume, period=3)["trend"] == "bullish"
    assert force_index(series(13, 12, 11, 10), volume, period=3)["trend"] == "bearish"


def test_force_index_needs_enough_rows():
    with pytest.raises(ValueError, match="at least 14"):
        force_index(series(1, 2, 3), series(1, 1, 1), period=13)


def test_bollinger_neutral():
    result = bollinger_bands(series(1, 2, 3, 4, 5), window=3, num_std=2)
    std = pstdev([3, 4, 5])
    assert result["middle"] == pytest.approx(4)
    assert result["upper"] == pytest.approx(4 + 2 * std)
    assert result["lower"] == pytest.approx(4 - 2 * std)
    assert result["percent_b"] == pytest.approx((5 - (4 - 2 * std)) / (4 * std))
    assert result["signal"] == "neutral"


def test_bollinger_overbought_and_oversold():
    assert bollinger_bands(series(10, 10, 10, 10, 20), window=3, num_std=1)["signal"] == "overbought"
    assert bollinger_bands(series(20, 20, 20, 20, 10), window=3, num_std=1)["signal"] == "oversold"


def test_bollinger_flat_series():
    result = bollinger_bands(series(5, 5, 5, 5, 5), window=3)
    assert result["percent_b"] == 0.5
    assert result["bandwidth"] == 0
    assert result["signal"] == "neutral"


def test_bollinger_needs_enough_rows():
    with pytest.raises(ValueError, match="at least 20"):
        bollinger_bands(series(1, 2, 3))


TODAY = pd.Timestamp("2026-10-06")


def test_insider_summary_counts_window_only():
    df = pd.DataFrame(
        {
            "Shares": [1000, 300, 50, 999],
            "Value": [100_000.0, 50_000.0, 0.0, 10.0],
            "Text": ["Purchase at price 100.00 per share.", "Sale at price 166.00 per share.", "Stock Gift", "Sale"],
            "Insider": ["A", "B", "C", "D"],
            "Position": ["CEO", "CFO", "Director", "Director"],
            "Start Date": pd.to_datetime(["2026-09-01", "2026-09-15", "2026-09-20", "2026-01-01"]),
        }
    )
    result = summarize_insider_transactions(df, days=90, today=TODAY)
    assert result["buy_count"] == 1
    assert result["sell_count"] == 1
    assert result["other_count"] == 1
    assert result["net_shares"] == 700
    assert result["buy_value"] == 100_000
    assert result["sell_value"] == 50_000
    assert result["top_transactions"][0] == {
        "date": "2026-09-01",
        "insider": "A",
        "position": "CEO",
        "text": "Purchase at price 100.00 per share.",
        "shares": 1000,
        "value": 100_000.0,
    }
    assert len(result["top_transactions"]) == 3


def test_insider_summary_handles_missing_data():
    for missing in (None, pd.DataFrame()):
        result = summarize_insider_transactions(missing, days=90, today=TODAY)
        assert result["buy_count"] == result["sell_count"] == result["net_shares"] == 0
        assert result["top_transactions"] == []
