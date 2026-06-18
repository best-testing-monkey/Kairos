"""Unit tests for kairos.adapter (KAI-2)."""
import numpy as np
import pandas as pd
import pytest

from kairos.adapter import to_kronos_frame
from kairos.errors import DataQualityError, InsufficientDataError

TZ = "America/New_York"


def _make_frame(n: int, extra_cols: bool = False) -> pd.DataFrame:
    """Build a synthetic price_cache-style frame."""
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz=TZ)
    data = {
        "Open": np.random.uniform(100, 200, n).astype(float),
        "High": np.random.uniform(200, 300, n).astype(float),
        "Low": np.random.uniform(50, 100, n).astype(float),
        "Close": np.random.uniform(100, 200, n).astype(float),
        "Volume": np.random.randint(1_000, 1_000_000, n).astype(float),
        "Dividends": np.zeros(n),
        "Stock Splits": np.zeros(n),
        "market_cap": np.zeros(n),
    }
    if extra_cols:
        data["SomeExtra"] = 0.0
    return pd.DataFrame(data, index=idx)


class TestRenameAndDtype:
    def test_columns_renamed(self):
        df = _make_frame(10)
        out, _ = to_kronos_frame(df, 10)
        assert set(out.columns) == {"open", "high", "low", "close", "volume"}

    def test_extra_columns_dropped(self):
        df = _make_frame(10, extra_cols=True)
        out, _ = to_kronos_frame(df, 10)
        assert "SomeExtra" not in out.columns

    def test_dtypes_float64(self):
        df = _make_frame(10)
        out, _ = to_kronos_frame(df, 10)
        for col in ["open", "high", "low", "close", "volume"]:
            assert out[col].dtype == np.float64

    def test_index_reset(self):
        df = _make_frame(10)
        out, _ = to_kronos_frame(df, 10)
        assert list(out.index) == list(range(10))


class TestAmount:
    def test_omit_default(self):
        df = _make_frame(5)
        out, _ = to_kronos_frame(df, 5, amount="omit")
        assert "amount" not in out.columns

    def test_close_volume(self):
        df = _make_frame(5)
        out, _ = to_kronos_frame(df, 5, amount="close_volume")
        assert "amount" in out.columns
        expected = out["close"] * out["volume"]
        pd.testing.assert_series_equal(out["amount"], expected, check_names=False)

    def test_auto_fallback_to_close_volume(self):
        df = _make_frame(5)
        out, _ = to_kronos_frame(df, 5, amount="auto")
        assert "amount" in out.columns
        expected = out["close"] * out["volume"]
        pd.testing.assert_series_equal(out["amount"], expected, check_names=False)

    def test_auto_prefers_native_amount(self):
        df = _make_frame(5)
        df["amount"] = 999.0
        out, _ = to_kronos_frame(df, 5, amount="auto")
        assert (out["amount"] == 999.0).all()

    def test_auto_prefers_vwap(self):
        df = _make_frame(5)
        df["vwap"] = 42.0
        out, _ = to_kronos_frame(df, 5, amount="auto")
        expected = 42.0 * df["Volume"].values
        np.testing.assert_allclose(out["amount"].values, expected)


class TestErrors:
    def test_insufficient_data_raises(self):
        df = _make_frame(5)
        with pytest.raises(InsufficientDataError) as exc_info:
            to_kronos_frame(df, 10)
        assert exc_info.value.have == 5
        assert exc_info.value.want == 10

    def test_nan_raises_data_quality_error(self):
        df = _make_frame(5)
        df.iloc[2, df.columns.get_loc("Close")] = float("nan")
        with pytest.raises(DataQualityError):
            to_kronos_frame(df, 5)


class TestTimestamp:
    def test_x_timestamp_length(self):
        df = _make_frame(10)
        _, x_ts = to_kronos_frame(df, 10)
        assert len(x_ts) == 10

    def test_x_timestamp_values_match_index(self):
        df = _make_frame(10)
        _, x_ts = to_kronos_frame(df, 10)
        pd.testing.assert_index_equal(pd.DatetimeIndex(x_ts), df.index, check_names=False)

    def test_tails_to_lookback(self):
        df = _make_frame(20)
        out, x_ts = to_kronos_frame(df, 10)
        assert len(out) == 10
        assert len(x_ts) == 10
        pd.testing.assert_index_equal(pd.DatetimeIndex(x_ts), df.index[-10:], check_names=False)
