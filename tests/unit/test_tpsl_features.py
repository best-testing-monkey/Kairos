"""E18-S02: Tests for price-history-derived TP/SL feature extraction.

No-lookahead invariant: extract_features() output is byte-identical whether
history is exactly truncated at `as_of` or has extra rows appended after it.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import math
import numpy as np
import pandas as pd
import pytest

from kairos_tpsl_features import extract_features, _compute_atr
from kairos_volatility import atr as volatility_atr


def _frame(closes, start="2024-01-01"):
    """Create a synthetic OHLCV DataFrame with realistic OHLC values."""
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=idx,
    )


def _historical_prices():
    """Create a long, realistic price history for feature extraction tests."""
    # 50 bars of price data, roughly trending
    closes = [100.0 + i * 0.5 + np.sin(i * 0.3) for i in range(50)]
    return _frame(closes)


class TestExtractFeaturesNoLookahead:
    """No-lookahead invariant: output unchanged by appending future bars."""

    def test_no_lookahead_at_position_30(self):
        """Features identical when history is truncated at bar 30 vs. extended."""
        full_history = _historical_prices()
        date_at_30 = full_history.index[30]

        # Truncate at position 30
        history_truncated = full_history[full_history.index <= date_at_30]

        # Extended version with extra bars after
        history_extended = full_history.copy()

        features_truncated = extract_features(
            ticker="AAPL",
            as_of=date_at_30.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history_truncated,
        )

        features_extended = extract_features(
            ticker="AAPL",
            as_of=date_at_30.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history_extended,
        )

        # All numeric values must be byte-identical
        for key in ["atr", "realized_vol", "trend_10", "range_position"]:
            assert features_truncated[key] == features_extended[key], (
                f"Lookahead leak on {key}: truncated={features_truncated[key]}, "
                f"extended={features_extended[key]}"
            )

        # String keys must match too
        assert features_truncated["asset_class"] == features_extended["asset_class"]
        assert features_truncated["interval"] == features_extended["interval"]

    def test_no_lookahead_at_position_20(self):
        """Second cutoff: no-lookahead at bar 20."""
        full_history = _historical_prices()
        date_at_20 = full_history.index[20]

        history_truncated = full_history[full_history.index <= date_at_20]
        history_extended = full_history.copy()

        features_truncated = extract_features(
            ticker="BTC-USD",
            as_of=date_at_20.strftime("%Y-%m-%d"),
            interval="1h",
            entry=50000.0,
            history=history_truncated,
        )

        features_extended = extract_features(
            ticker="BTC-USD",
            as_of=date_at_20.strftime("%Y-%m-%d"),
            interval="1h",
            entry=50000.0,
            history=history_extended,
        )

        for key in ["atr", "realized_vol", "trend_10", "range_position"]:
            assert features_truncated[key] == features_extended[key]

        assert features_truncated["asset_class"] == features_extended["asset_class"]
        assert features_truncated["interval"] == features_extended["interval"]


class TestATRCalculation:
    """ATR calculation matches ATRBracketStrategy's own ATR."""

    def test_atr_matches_volatility_atr(self):
        """_compute_atr produces same value as kairos_volatility.atr()."""
        history = _historical_prices()

        our_atr = _compute_atr(history, n=14)
        their_atr = volatility_atr(history, n=14)

        assert float(our_atr) == float(their_atr), (
            f"ATR mismatch: _compute_atr={our_atr}, "
            f"kairos_volatility.atr={their_atr}"
        )

    def test_atr_included_in_features(self):
        """Returned features dict includes ATR key with correct value."""
        history = _historical_prices()
        date = history.index[-1]

        features = extract_features(
            ticker="XYZ",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history,
        )

        assert "atr" in features
        expected_atr = volatility_atr(history, n=14)
        assert features["atr"] == float(expected_atr)


class TestAssetClassFeature:
    """Asset class is correctly determined and included."""

    def test_asset_class_for_equity(self):
        """Equity ticker classified correctly."""
        history = _historical_prices()
        date = history.index[-1]

        features = extract_features(
            ticker="AAPL",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history,
        )

        assert features["asset_class"] == "equity"

    def test_asset_class_for_crypto(self):
        """Crypto ticker (ending in -USD) classified correctly."""
        history = _historical_prices()
        date = history.index[-1]

        features = extract_features(
            ticker="BTC-USD",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=50000.0,
            history=history,
        )

        assert features["asset_class"] == "crypto"

    def test_asset_class_for_fx(self):
        """FX ticker (ending in =X) classified correctly."""
        history = _historical_prices()
        date = history.index[-1]

        features = extract_features(
            ticker="EUR=X",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=1.1,
            history=history,
        )

        assert features["asset_class"] == "fx"

    def test_asset_class_for_commodity(self):
        """Commodity ticker (ending in =F) classified correctly."""
        history = _historical_prices()
        date = history.index[-1]

        features = extract_features(
            ticker="CL=F",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history,
        )

        assert features["asset_class"] == "commodity"


class TestInsufficientHistory:
    """Proper error handling when history is too short."""

    def test_raises_on_insufficient_history_less_than_21(self):
        """Raises ValueError when history has fewer than 21 bars."""
        short_history = _frame([100.0, 101.0, 102.0])  # Only 3 bars

        with pytest.raises(ValueError, match="Insufficient history"):
            extract_features(
                ticker="TEST",
                as_of="2024-01-03",
                interval="1d",
                entry=101.0,
                history=short_history,
            )

    def test_raises_on_exactly_20_bars(self):
        """Raises ValueError when history has exactly 20 bars (need 21)."""
        history = _frame([100.0 + i for i in range(20)])  # Exactly 20 bars

        with pytest.raises(ValueError, match="Insufficient history"):
            extract_features(
                ticker="TEST",
                as_of=history.index[-1].strftime("%Y-%m-%d"),
                interval="1d",
                entry=105.0,
                history=history,
            )

    def test_accepts_exactly_21_bars(self):
        """Accepts when history has exactly 21 bars."""
        history = _frame([100.0 + i for i in range(21)])  # Exactly 21 bars
        date = history.index[-1]

        # Should not raise
        features = extract_features(
            ticker="TEST",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=110.0,
            history=history,
        )

        assert "atr" in features
        assert "realized_vol" in features


class TestFeatureBounds:
    """Features are in reasonable ranges and match expected behavior."""

    def test_realized_vol_is_positive(self):
        """Realized volatility should be positive (stdev of returns)."""
        history = _historical_prices()
        date = history.index[-1]

        features = extract_features(
            ticker="TEST",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history,
        )

        assert features["realized_vol"] >= 0.0

    def test_range_position_in_bounds(self):
        """Range position should be in [0, 1] when entry is within range."""
        history = _frame([100.0] * 25)  # Flat price
        date = history.index[-1]

        features = extract_features(
            ticker="TEST",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history,
        )

        assert 0.0 <= features["range_position"] <= 1.0

    def test_interval_passthrough(self):
        """Interval is passed through as-is."""
        history = _historical_prices()
        date = history.index[-1]

        for interval_val in ["1d", "1h", "4h", "15m"]:
            features = extract_features(
                ticker="TEST",
                as_of=date.strftime("%Y-%m-%d"),
                interval=interval_val,
                entry=100.0,
                history=history,
            )

            assert features["interval"] == interval_val

    def test_trend_captures_direction(self):
        """Trend is positive when price rises, negative when falls."""
        # Uptrend: prices gradually increase
        uptrend_history = _frame([100.0 + i * 2 for i in range(50)])
        date_up = uptrend_history.index[-1]
        features_up = extract_features(
            ticker="TEST",
            as_of=date_up.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=uptrend_history,
        )
        assert features_up["trend_10"] > 0

        # Downtrend: prices gradually decrease
        downtrend_history = _frame([150.0 - i * 2 for i in range(50)])
        date_down = downtrend_history.index[-1]
        features_down = extract_features(
            ticker="TEST",
            as_of=date_down.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=downtrend_history,
        )
        assert features_down["trend_10"] < 0


class TestReturnDictShape:
    """Feature dict has expected structure and types."""

    def test_returns_dict_with_all_keys(self):
        """Returned dict has all expected keys."""
        history = _historical_prices()
        date = history.index[-1]

        features = extract_features(
            ticker="TEST",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history,
        )

        expected_keys = {
            "atr",
            "realized_vol",
            "asset_class",
            "interval",
            "trend_10",
            "range_position",
        }
        assert set(features.keys()) == expected_keys

    def test_numeric_values_are_floats(self):
        """Numeric features are float types."""
        history = _historical_prices()
        date = history.index[-1]

        features = extract_features(
            ticker="TEST",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history,
        )

        numeric_keys = ["atr", "realized_vol", "trend_10", "range_position"]
        for key in numeric_keys:
            assert isinstance(features[key], (float, np.floating))

    def test_string_features_are_strings(self):
        """String features are str types."""
        history = _historical_prices()
        date = history.index[-1]

        features = extract_features(
            ticker="TEST",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history,
        )

        assert isinstance(features["asset_class"], str)
        assert isinstance(features["interval"], str)


class TestTimezoneHandling:
    """E21-S04: Timezone-aware history handling (no crash on tz-aware index)."""

    def test_tz_aware_history_no_crash(self):
        """History with tz-aware index does not crash on comparison."""
        history_naive = _historical_prices()
        # Convert to tz-aware (America/New_York)
        history_aware = history_naive.copy()
        history_aware.index = history_aware.index.tz_localize("America/New_York")

        date = history_aware.index[30]
        as_of_str = date.strftime("%Y-%m-%d")

        # Should not raise TypeError on tz-aware comparison
        features = extract_features(
            ticker="AAPL",
            as_of=as_of_str,
            interval="1d",
            entry=100.0,
            history=history_aware,
        )

        assert "atr" in features
        assert "realized_vol" in features

    def test_tz_aware_no_lookahead_invariant(self):
        """No-lookahead invariant preserved for tz-aware history."""
        history_naive = _historical_prices()
        # Convert to tz-aware (UTC)
        history_aware = history_naive.copy()
        history_aware.index = history_aware.index.tz_localize("UTC")

        date_aware = history_aware.index[30]
        as_of_str = date_aware.strftime("%Y-%m-%d")

        # Truncate at position 30
        history_truncated = history_aware[history_aware.index <= date_aware]

        # Extended version with extra bars
        history_extended = history_aware.copy()

        features_truncated = extract_features(
            ticker="AAPL",
            as_of=as_of_str,
            interval="1d",
            entry=100.0,
            history=history_truncated,
        )

        features_extended = extract_features(
            ticker="AAPL",
            as_of=as_of_str,
            interval="1d",
            entry=100.0,
            history=history_extended,
        )

        # All numeric values must be byte-identical
        for key in ["atr", "realized_vol", "trend_10", "range_position"]:
            assert features_truncated[key] == features_extended[key], (
                f"Lookahead leak on tz-aware with {key}: "
                f"truncated={features_truncated[key]}, "
                f"extended={features_extended[key]}"
            )


class TestFlatRangeGuard:
    """E21-S04: Division-by-zero guard for flat 20-bar range."""

    def test_flat_range_returns_sentinel(self):
        """Flat range (high==low) returns 0.5, not NaN or inf."""
        # Create 25 bars with truly identical OHLC (high == low == close)
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        flat_history = pd.DataFrame(
            {
                "open": [100.0] * 25,
                "high": [100.0] * 25,
                "low": [100.0] * 25,
                "close": [100.0] * 25,
                "volume": [1000.0] * 25,
            },
            index=idx,
        )
        date = flat_history.index[-1]

        features = extract_features(
            ticker="TEST",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=flat_history,
        )

        # Should return 0.5 (sentinel for degenerate range)
        assert features["range_position"] == 0.5
        # Verify it's not NaN or inf
        assert not math.isnan(features["range_position"])
        assert not math.isinf(features["range_position"])

    def test_near_flat_range_defined_value(self):
        """Near-flat range (diff < epsilon) returns 0.5."""
        # Create history with near-identical OHLC (within epsilon of zero range)
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        # Create values where high - low is smaller than epsilon
        near_flat_history = pd.DataFrame(
            {
                "open": [100.0 + 1e-10] * 25,
                "high": [100.0 + 2e-10] * 25,  # Difference of 1e-10 from open
                "low": [100.0] * 25,  # Difference of 1e-10 from open
                "close": [100.0 + 1e-10] * 25,
                "volume": [1000.0] * 25,
            },
            index=idx,
        )
        date = near_flat_history.index[-1]

        features = extract_features(
            ticker="TEST",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=near_flat_history,
        )

        # Should return 0.5 (degenerate range)
        assert features["range_position"] == 0.5
        assert not math.isnan(features["range_position"])
        assert not math.isinf(features["range_position"])

    def test_normal_range_unchanged(self):
        """Normal (non-flat) range still computes correctly."""
        history = _historical_prices()
        date = history.index[-1]

        features = extract_features(
            ticker="TEST",
            as_of=date.strftime("%Y-%m-%d"),
            interval="1d",
            entry=100.0,
            history=history,
        )

        # Normal range should be in [0, 1]
        assert 0.0 <= features["range_position"] <= 1.0
        # And should NOT be the sentinel value 0.5 (extremely unlikely to be exact)
        # (only checking it's not obviously pathological)
        assert not math.isnan(features["range_position"])
        assert not math.isinf(features["range_position"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
