"""KAI-6: Contract tests for KronosPredictor.predict input.

These tests exercise the full get_forecast_window pipeline against a mocked
price_cache and assert that the returned tuple satisfies every constraint the
model's predict() method imposes.  No network calls, no model download.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import kairos
from kairos.errors import NoDataError, UnsupportedIntervalError

TZ = "America/New_York"
CAL = "XNYS"


def _make_ohlcv_frame(n: int, tz: str = TZ) -> pd.DataFrame:
    """Return a synthetic price_cache-style DataFrame with n rows."""
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz=tz)
    rng = np.random.default_rng(42)
    closes = 100 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "Open": closes + rng.uniform(-1, 1, n),
            "High": closes + rng.uniform(0, 2, n),
            "Low": closes - rng.uniform(0, 2, n),
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
    kairos.configure(remote=False, calendar=CAL, tz=TZ)


LOOKBACK = 30
PRED_LEN = 5


def _run(lookback=LOOKBACK, pred_len=PRED_LEN, amount="omit"):
    frame = _make_ohlcv_frame(lookback + 20)  # over-fetch so tail works
    with patch("price_cache.get_price_data", return_value=frame):
        return kairos.get_forecast_window(
            "AAPL", "1d", lookback, pred_len,
            end="2024-02-28",
            amount=amount,
        )


class TestPredictContract:
    """All assertions mirror the KronosPredictor.predict input spec."""

    def test_returns_three_tuple(self):
        result = _run()
        assert isinstance(result, tuple) and len(result) == 3

    def test_x_df_column_set_exact(self):
        x_df, _, _ = _run()
        assert set(x_df.columns) == {"open", "high", "low", "close", "volume"}

    def test_x_df_column_set_with_amount(self):
        x_df, _, _ = _run(amount="close_volume")
        assert set(x_df.columns) == {"open", "high", "low", "close", "volume", "amount"}

    def test_x_df_length_equals_lookback(self):
        x_df, _, _ = _run()
        assert len(x_df) == LOOKBACK

    def test_x_timestamp_length_equals_x_df(self):
        x_df, x_ts, _ = _run()
        assert len(x_ts) == len(x_df)

    def test_y_timestamp_length_equals_pred_len(self):
        _, _, y_ts = _run()
        assert len(y_ts) == PRED_LEN

    def test_x_timestamp_strictly_increasing(self):
        _, x_ts, _ = _run()
        diffs = pd.Series(pd.DatetimeIndex(x_ts)).diff().iloc[1:]
        assert (diffs > pd.Timedelta(0)).all()

    def test_y_timestamp_strictly_increasing(self):
        _, _, y_ts = _run()
        diffs = pd.Series(pd.DatetimeIndex(y_ts)).diff().iloc[1:]
        assert (diffs > pd.Timedelta(0)).all()

    def test_y_first_after_x_last(self):
        _, x_ts, y_ts = _run()
        assert y_ts.iloc[0] > x_ts.iloc[-1]

    def test_same_tz_across_series(self):
        _, x_ts, y_ts = _run()
        x_tz = str(pd.DatetimeIndex(x_ts).tz)
        y_tz = str(pd.DatetimeIndex(y_ts).tz)
        assert x_tz == y_tz

    def test_x_df_dtypes_float64(self):
        x_df, _, _ = _run()
        for col in ["open", "high", "low", "close", "volume"]:
            assert x_df[col].dtype == np.float64, f"{col} not float64"

    def test_x_df_index_is_range(self):
        x_df, _, _ = _run()
        assert list(x_df.index) == list(range(LOOKBACK))

    def test_no_nans_in_x_df(self):
        x_df, _, _ = _run()
        assert not x_df[["open", "high", "low", "close", "volume"]].isna().any().any()


class TestErrors:
    def test_none_data_raises_no_data_error(self):
        with patch("price_cache.get_price_data", return_value=None):
            with pytest.raises(NoDataError):
                kairos.get_forecast_window("DEAD", "1d", 10, 5, end="2024-02-28")

    def test_unsupported_interval_raises(self):
        with pytest.raises(UnsupportedIntervalError):
            kairos.get_forecast_window("AAPL", "99x", 10, 5, end="2024-02-28")

    def test_various_lookback_pred_len_combinations(self):
        for lookback, pred_len in [(10, 1), (50, 10), (5, 20)]:
            frame = _make_ohlcv_frame(lookback + 20)
            with patch("price_cache.get_price_data", return_value=frame):
                x_df, x_ts, y_ts = kairos.get_forecast_window(
                    "AAPL", "1d", lookback, pred_len, end="2024-06-14"
                )
            assert len(x_df) == lookback
            assert len(y_ts) == pred_len
