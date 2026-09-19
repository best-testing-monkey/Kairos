"""
Tests for kairos_prediction_usage.py

Tests cover:
- Synthetic bar value correctness (OHLC median percentiles, current_price entry)
- Volume carry-forward from last real bar
- Timestamp advancement for different interval values
- No mutation of input AssetPrediction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from kairos_prediction_usage import _build_synthetic_bar, _interval_to_timedelta, distribution_as_bar


@dataclass
class MockDistribution:
    """Mock KairosDistribution with controlled stats for testing."""
    stats: dict
    df: pd.DataFrame = None


@dataclass
class MockAssetPrediction:
    """Mock AssetPrediction for testing."""
    symbol: str
    dist: MockDistribution
    current_price: float
    history: pd.DataFrame


class TestIntervalToTimedelta:
    """Tests for _interval_to_timedelta helper."""

    def test_converts_1_minute(self):
        result = _interval_to_timedelta("1m")
        assert result == timedelta(minutes=1)

    def test_converts_60_minutes(self):
        result = _interval_to_timedelta("60m")
        assert result == timedelta(minutes=60)

    def test_converts_1_hour(self):
        result = _interval_to_timedelta("1h")
        assert result == timedelta(hours=1)

    def test_converts_4_hours(self):
        result = _interval_to_timedelta("4h")
        assert result == timedelta(hours=4)

    def test_converts_1_day(self):
        result = _interval_to_timedelta("1d")
        assert result == timedelta(days=1)

    def test_converts_5_days(self):
        result = _interval_to_timedelta("5d")
        assert result == timedelta(days=5)

    def test_converts_1_week(self):
        result = _interval_to_timedelta("1wk")
        assert result == timedelta(weeks=1)

    def test_rejects_monthly_interval(self):
        try:
            _interval_to_timedelta("1mo")
            assert False, "Should raise ValueError for '1mo'"
        except ValueError as e:
            assert "Cannot convert interval" in str(e)

    def test_rejects_invalid_format(self):
        try:
            _interval_to_timedelta("invalid")
            assert False, "Should raise ValueError for invalid format"
        except ValueError as e:
            assert "Cannot convert interval" in str(e)


class TestBuildSyntheticBar:
    """Tests for _build_synthetic_bar function."""

    def _make_history_df(self, base_ts, num_rows=10):
        """Create a mock history DataFrame."""
        dates = pd.date_range(base_ts, periods=num_rows, freq="D")
        return pd.DataFrame(
            {
                "open": np.arange(100.0, 100.0 + num_rows),
                "high": np.arange(101.0, 101.0 + num_rows),
                "low": np.arange(99.0, 99.0 + num_rows),
                "close": np.arange(100.5, 100.5 + num_rows),
                "volume": np.full(num_rows, 1000000.0),
            },
            index=dates,
        )

    def _make_prediction(
        self, base_ts="2026-01-01", current_price=50000.0,
        open_pct=50000.0, high_pct=51000.0, low_pct=49000.0, close_pct=50500.0,
        last_volume=2000000.0
    ):
        """Helper to create a mock AssetPrediction with controlled stats."""
        history = self._make_history_df(base_ts)
        history.iloc[-1, history.columns.get_loc("volume")] = last_volume

        dist = MockDistribution(
            stats={
                "open": {"pct_50": open_pct},
                "high": {"pct_50": high_pct},
                "low": {"pct_50": low_pct},
                "close": {"pct_50": close_pct},
            },
            df=pd.DataFrame(),  # not used in _build_synthetic_bar
        )

        return MockAssetPrediction(
            symbol="BTC-USD",
            dist=dist,
            current_price=current_price,
            history=history,
        )

    def test_open_equals_current_price(self):
        """Synthetic bar's open should equal current_price."""
        pred = self._make_prediction(current_price=50000.0)
        bar = _build_synthetic_bar(pred, interval="1d")
        assert bar["open"] == 50000.0

    def test_high_equals_pct_50(self):
        """Synthetic bar's high should equal dist.stats["high"]["pct_50"]."""
        pred = self._make_prediction(high_pct=51000.0)
        bar = _build_synthetic_bar(pred, interval="1d")
        assert bar["high"] == 51000.0

    def test_low_equals_pct_50(self):
        """Synthetic bar's low should equal dist.stats["low"]["pct_50"]."""
        pred = self._make_prediction(low_pct=49000.0)
        bar = _build_synthetic_bar(pred, interval="1d")
        assert bar["low"] == 49000.0

    def test_close_equals_pct_50(self):
        """Synthetic bar's close should equal dist.stats["close"]["pct_50"]."""
        pred = self._make_prediction(close_pct=50500.0)
        bar = _build_synthetic_bar(pred, interval="1d")
        assert bar["close"] == 50500.0

    def test_volume_carries_forward(self):
        """Synthetic bar's volume should equal last real bar's volume."""
        pred = self._make_prediction(last_volume=2000000.0)
        bar = _build_synthetic_bar(pred, interval="1d")
        assert bar["volume"] == 2000000.0

    def test_volume_different_from_second_to_last(self):
        """Confirm volume uses last row, not penultimate."""
        history = self._make_history_df("2026-01-01", num_rows=10)
        # Set different volumes for last two rows
        history.iloc[-2, history.columns.get_loc("volume")] = 1000000.0
        history.iloc[-1, history.columns.get_loc("volume")] = 5000000.0

        dist = MockDistribution(
            stats={
                "open": {"pct_50": 100.0},
                "high": {"pct_50": 101.0},
                "low": {"pct_50": 99.0},
                "close": {"pct_50": 100.5},
            },
            df=pd.DataFrame(),
        )

        pred = MockAssetPrediction(
            symbol="BTC-USD",
            dist=dist,
            current_price=100.0,
            history=history,
        )

        bar = _build_synthetic_bar(pred, interval="1d")
        assert bar["volume"] == 5000000.0

    def test_timestamp_advances_by_one_day(self):
        """Synthetic bar timestamp should be exactly one day after last real bar."""
        base_ts = "2026-01-01 12:00:00"
        pred = self._make_prediction(base_ts=base_ts)
        bar = _build_synthetic_bar(pred, interval="1d")

        last_ts = pred.history.index[-1]
        expected_ts = last_ts + timedelta(days=1)
        assert bar.name == expected_ts

    def test_timestamp_advances_by_one_hour(self):
        """Synthetic bar timestamp should be exactly one hour after last real bar."""
        dates = pd.date_range("2026-01-01", periods=10, freq="h")
        history = pd.DataFrame(
            {
                "open": np.arange(100.0, 110.0),
                "high": np.arange(101.0, 111.0),
                "low": np.arange(99.0, 109.0),
                "close": np.arange(100.5, 110.5),
                "volume": np.full(10, 1000000.0),
            },
            index=dates,
        )

        dist = MockDistribution(
            stats={
                "open": {"pct_50": 100.0},
                "high": {"pct_50": 101.0},
                "low": {"pct_50": 99.0},
                "close": {"pct_50": 100.5},
            },
            df=pd.DataFrame(),
        )

        pred = MockAssetPrediction(
            symbol="BTC-USD",
            dist=dist,
            current_price=100.0,
            history=history,
        )

        bar = _build_synthetic_bar(pred, interval="1h")

        last_ts = pred.history.index[-1]
        expected_ts = last_ts + timedelta(hours=1)
        assert bar.name == expected_ts

    def test_timestamp_advances_by_four_hours(self):
        """Synthetic bar timestamp should be exactly 4 hours after last real bar."""
        base_ts = "2026-01-01 12:00:00"
        dates = pd.date_range(base_ts, periods=10, freq="4h")
        history = pd.DataFrame(
            {
                "open": np.arange(100.0, 110.0),
                "high": np.arange(101.0, 111.0),
                "low": np.arange(99.0, 109.0),
                "close": np.arange(100.5, 110.5),
                "volume": np.full(10, 1000000.0),
            },
            index=dates,
        )

        dist = MockDistribution(
            stats={
                "open": {"pct_50": 100.0},
                "high": {"pct_50": 101.0},
                "low": {"pct_50": 99.0},
                "close": {"pct_50": 100.5},
            },
            df=pd.DataFrame(),
        )

        pred = MockAssetPrediction(
            symbol="BTC-USD",
            dist=dist,
            current_price=100.0,
            history=history,
        )

        bar = _build_synthetic_bar(pred, interval="4h")

        last_ts = pred.history.index[-1]
        expected_ts = last_ts + timedelta(hours=4)
        assert bar.name == expected_ts

    def test_bar_has_all_columns(self):
        """Synthetic bar should have all required OHLCV columns."""
        pred = self._make_prediction()
        bar = _build_synthetic_bar(pred, interval="1d")
        assert "open" in bar.index
        assert "high" in bar.index
        assert "low" in bar.index
        assert "close" in bar.index
        assert "volume" in bar.index

    def test_input_not_mutated(self):
        """Input AssetPrediction and its history should not be mutated."""
        pred = self._make_prediction()
        original_history = pred.history.copy()
        original_current_price = pred.current_price
        original_dist_stats = {k: v.copy() for k, v in pred.dist.stats.items()}

        _build_synthetic_bar(pred, interval="1d")

        # Check history unchanged
        assert pred.history.equals(original_history)
        # Check current_price unchanged
        assert pred.current_price == original_current_price
        # Check dist.stats unchanged
        assert pred.dist.stats == original_dist_stats
        # Check history length unchanged (not appended to)
        assert len(pred.history) == len(original_history)

    def test_returned_series_is_series(self):
        """Returned value should be a pd.Series."""
        pred = self._make_prediction()
        bar = _build_synthetic_bar(pred, interval="1d")
        assert isinstance(bar, pd.Series)

    def test_multiple_calls_independent(self):
        """Multiple calls with different intervals should not interfere."""
        pred = self._make_prediction()

        bar_1d = _build_synthetic_bar(pred, interval="1d")
        bar_1h = _build_synthetic_bar(pred, interval="1h")

        assert bar_1d.name != bar_1h.name
        # 1d should be 24 hours later than 1h
        time_diff = bar_1d.name - bar_1h.name
        assert time_diff == timedelta(days=1) - timedelta(hours=1)


class TestDistributionAsBar:
    """Tests for distribution_as_bar function."""

    def _make_history_df(self, base_ts, num_rows=10):
        """Create a mock history DataFrame."""
        dates = pd.date_range(base_ts, periods=num_rows, freq="D")
        return pd.DataFrame(
            {
                "open": np.arange(100.0, 100.0 + num_rows),
                "high": np.arange(101.0, 101.0 + num_rows),
                "low": np.arange(99.0, 99.0 + num_rows),
                "close": np.arange(100.5, 100.5 + num_rows),
                "volume": np.full(num_rows, 1000000.0),
            },
            index=dates,
        )

    def _make_prediction(
        self, base_ts="2026-01-01", current_price=50000.0,
        open_pct=50000.0, high_pct=51000.0, low_pct=49000.0, close_pct=50500.0
    ):
        """Helper to create a mock AssetPrediction with controlled stats."""
        history = self._make_history_df(base_ts)

        dist = MockDistribution(
            stats={
                "open": {"pct_50": open_pct},
                "high": {"pct_50": high_pct},
                "low": {"pct_50": low_pct},
                "close": {"pct_50": close_pct},
            },
            df=pd.DataFrame(),
        )

        return MockAssetPrediction(
            symbol="BTC-USD",
            dist=dist,
            current_price=current_price,
            history=history,
        )

    def test_current_price_unchanged(self):
        """distribution_as_bar should not modify current_price."""
        pred = self._make_prediction(current_price=50000.0)
        result = distribution_as_bar(pred, interval="1d")
        assert result.current_price == pred.current_price
        assert result.current_price == 50000.0

    def test_dist_unchanged(self):
        """distribution_as_bar should not modify dist."""
        pred = self._make_prediction()
        result = distribution_as_bar(pred, interval="1d")
        assert result.dist is pred.dist

    def test_symbol_unchanged(self):
        """distribution_as_bar should not modify symbol."""
        pred = self._make_prediction()
        result = distribution_as_bar(pred, interval="1d")
        assert result.symbol == pred.symbol
        assert result.symbol == "BTC-USD"

    def test_history_one_row_added(self):
        """Returned history should have exactly one more row than input."""
        pred = self._make_prediction()
        original_len = len(pred.history)
        result = distribution_as_bar(pred, interval="1d")
        assert len(result.history) == original_len + 1

    def test_original_history_rows_unchanged(self):
        """All original history rows should be identical in result."""
        pred = self._make_prediction()
        original_history = pred.history.copy()
        result = distribution_as_bar(pred, interval="1d")

        # Check first n-1 rows match
        for i in range(len(original_history)):
            pd.testing.assert_series_equal(
                result.history.iloc[i],
                original_history.iloc[i],
                check_names=False
            )

    def test_new_row_is_synthetic_bar(self):
        """The appended row should be the synthetic bar."""
        pred = self._make_prediction(
            high_pct=51000.0, low_pct=49000.0,
            close_pct=50500.0, current_price=50000.0
        )
        result = distribution_as_bar(pred, interval="1d")

        new_row = result.history.iloc[-1]
        assert new_row["open"] == 50000.0
        assert new_row["high"] == 51000.0
        assert new_row["low"] == 49000.0
        assert new_row["close"] == 50500.0

    def test_input_not_mutated(self):
        """Input AssetPrediction should not be mutated."""
        pred = self._make_prediction()
        original_history = pred.history.copy()
        original_len = len(pred.history)

        distribution_as_bar(pred, interval="1d")

        assert len(pred.history) == original_len
        pd.testing.assert_frame_equal(pred.history, original_history)

    def test_returned_is_asset_prediction(self):
        """Return value should be an AssetPrediction (or compatible)."""
        pred = self._make_prediction()
        result = distribution_as_bar(pred, interval="1d")
        assert hasattr(result, "symbol")
        assert hasattr(result, "dist")
        assert hasattr(result, "current_price")
        assert hasattr(result, "history")

    def test_different_intervals(self):
        """Should work with different interval strings."""
        pred = self._make_prediction()

        result_1d = distribution_as_bar(pred, interval="1d")
        result_1h = distribution_as_bar(pred, interval="1h")

        # Both should add exactly one row
        assert len(result_1d.history) == len(pred.history) + 1
        assert len(result_1h.history) == len(pred.history) + 1

        # But timestamps should be different
        new_ts_1d = result_1d.history.index[-1]
        new_ts_1h = result_1h.history.index[-1]
        assert new_ts_1d != new_ts_1h
