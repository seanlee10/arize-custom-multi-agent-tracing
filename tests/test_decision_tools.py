import pandas as pd
import pytest

from decision_agent import market_data
from decision_agent.tools import LOCAL_TOOLS, run_bollinger_bands, run_force_index, run_insider_trades


@pytest.fixture
def fake_prices(monkeypatch):
    closes = [100 + i for i in range(30)]
    history = pd.DataFrame({"Close": closes, "Volume": [1_000] * 30}, dtype=float)
    monkeypatch.setattr(market_data, "fetch_price_history", lambda ticker, period="6mo": history)


def test_run_force_index(fake_prices):
    result = run_force_index("AAPL")
    assert result["ticker"] == "AAPL"
    assert result["trend"] == "bullish"
    attrs = LOCAL_TOOLS["force_index"].span_attributes(result)
    assert attrs == {"finagent.force_index.ema": result["force_index_ema"], "finagent.force_index.trend": "bullish"}


def test_run_bollinger_bands(fake_prices):
    result = run_bollinger_bands("AAPL", window=20)
    assert result["ticker"] == "AAPL"
    attrs = LOCAL_TOOLS["bollinger_bands"].span_attributes(result)
    assert attrs["finagent.bollinger.signal"] == result["signal"]
    assert attrs["finagent.bollinger.percent_b"] == result["percent_b"]


def test_run_insider_trades_without_data(monkeypatch):
    monkeypatch.setattr(market_data, "fetch_insider_transactions", lambda ticker: None)
    result = run_insider_trades("SPY")
    assert result["ticker"] == "SPY"
    assert result["net_shares"] == 0
    assert LOCAL_TOOLS["insider_trades"].span_attributes(result) == {
        "finagent.insider.net_shares": 0,
        "finagent.insider.txn_count": 0,
    }


def test_tool_definitions_are_anthropic_shaped():
    assert set(LOCAL_TOOLS) == {"force_index", "bollinger_bands", "insider_trades"}
    for tool in LOCAL_TOOLS.values():
        definition = tool.definition()
        assert set(definition) == {"name", "description", "input_schema"}
        assert definition["input_schema"]["required"] == ["ticker"]
