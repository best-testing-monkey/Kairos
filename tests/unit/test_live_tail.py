"""KAI-7: Live-tail tests.

price_cache never caches or delivers incomplete (in-progress) bars, so
end=None (live mode) requires no special partial-bar handling.  These
tests confirm that the live path behaves identically to a historic call.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import kairos

TZ = "America/New_York"


def _make_frame(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz=TZ)
    rng = np.random.default_rng(7)
    closes = 100 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "Open": closes - 0.5,
            "High": closes + 1.0,
            "Low": closes - 1.0,
            "Close": closes,
            "Volume": rng.integers(100_000, 1_000_000, n).astype(float),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
            "market_cap": 0.0,
        },
        index=idx,
    )


@pytest.fixture(autouse=True)
def configure_kairos():
    kairos.configure(remote=False)


class TestLiveTail:
    def test_end_none_returns_valid_tuple(self):
        """end=None (live) produces the same shaped output as a past-date call."""
        frame = _make_frame(50)
        with patch("price_cache.get_price_data", return_value=frame):
            x_df, x_ts, y_ts = kairos.get_forecast_window(
                "AAPL", "1d", 20, 5, end=None
            )
        assert len(x_df) == 20
        assert len(x_ts) == 20
        assert len(y_ts) == 5

    def test_end_none_satisfies_contract(self):
        """Live call satisfies the full predict input contract."""
        frame = _make_frame(50)
        with patch("price_cache.get_price_data", return_value=frame):
            x_df, x_ts, y_ts = kairos.get_forecast_window(
                "AAPL", "1d", 20, 3, end=None
            )
        assert set(x_df.columns) == {"open", "high", "low", "close", "volume"}
        assert pd.DatetimeIndex(x_ts).is_monotonic_increasing
        assert pd.DatetimeIndex(y_ts).is_monotonic_increasing
        assert y_ts.iloc[0] > x_ts.iloc[-1]

    def test_live_and_historic_produce_same_structure(self):
        """Live and historic paths return identical column/dtype structure."""
        frame = _make_frame(50)
        with patch("price_cache.get_price_data", return_value=frame):
            x_live, _, _ = kairos.get_forecast_window(
                "AAPL", "1d", 20, 5, end=None
            )
            x_hist, _, _ = kairos.get_forecast_window(
                "AAPL", "1d", 20, 5, end="2024-06-14"
            )
        assert list(x_live.columns) == list(x_hist.columns)
        assert list(x_live.dtypes) == list(x_hist.dtypes)
