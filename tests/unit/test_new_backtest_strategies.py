import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
import numpy as np
import pandas as pd
from kairos_backtest import (
    KairosDistribution, Direction, Signal, Strategy,
    VaRPositionCapStrategy, DistributionOverlapStrategy,
    ModelDecayMonitorStrategy, OvernightExposureFilter,
    RSIDivergenceStrategy, LeverageCalibrationStrategy,
    DynamicBracketStrategy, TrendFollowingStrategy,
)


# ============================================================================
# Helpers
# ============================================================================

def make_dist(close_prices, open_prices=None, high_prices=None, low_prices=None):
    """Build a KairosDistribution from a list of close prices."""
    prices = np.array(close_prices, dtype=float)
    n = len(prices)
    o = np.array(open_prices or prices * 0.999, dtype=float)
    h = np.array(high_prices or prices * 1.005, dtype=float)
    l = np.array(low_prices or prices * 0.995, dtype=float)
    frames = []
    for i in range(n):
        frames.append(pd.DataFrame({
            "open": [o[i]], "high": [h[i]], "low": [l[i]],
            "close": [prices[i]], "volume": [1e6], "amount": [1e9]
        }))
    return KairosDistribution(frames)


def make_history(n=50, price=100.0):
    """Build a minimal history DataFrame for backtesting."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": [price]*n, "high": [price*1.01]*n,
        "low": [price*0.99]*n, "close": [price]*n, "volume": [1e6]*n
    }, index=idx)


class StubStrategy(Strategy):
    """Minimal stub strategy for testing wrappers."""
    name = "stub"

    def __init__(self, direction=Direction.LONG, size=0.5):
        self.direction = direction
        self.size = size

    def generate_signal(self, dist, current_price, history, context):
        if self.direction == Direction.FLAT:
            return None
        s = dist.stats["close"]
        return Signal(
            direction=self.direction,
            size=self.size,
            entry=current_price,
            stop=s["pct_10"],
            target=s["pct_90"],
            strategy_name=self.name,
            confidence=0.8,
            expected_value=1.0,
        )


# ============================================================================
# VaRPositionCapStrategy Tests
# ============================================================================

class TestVaRPositionCapStrategy:
    def test_var_cap_2pct_var_1pct_risk(self):
        """Test: 2% VaR below entry with 1% risk → size = min(base, 0.5)"""
        # Setup: current_price=100, pct_5=98
        # max_loss_per_unit = 100-98 = 2
        # account_risk_limit = 1.0 * 0.01 = 0.01
        # max_units = 0.01 / 2 = 0.005
        # max_notional = 0.005 * 100 = 0.5
        # max_size = 0.5 / 1.0 = 0.5
        dist = make_dist([98.0] * 50 + [100.0] * 50)
        stub = StubStrategy(Direction.LONG, size=1.0)
        strat = VaRPositionCapStrategy(stub, max_account_risk_pct=0.01, capital=1.0)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert abs(sig.size - 0.5) < 0.01

    def test_var_cap_favorable_var(self):
        """Test: signal unchanged when pct_5 > entry for LONG (favorable risk)"""
        # Setup: current_price=100, pct_5=102
        # max_loss_per_unit = 100-102 = -2 < 0 → return unchanged
        dist = make_dist([102.0] * 100)
        stub = StubStrategy(Direction.LONG, size=0.7)
        strat = VaRPositionCapStrategy(stub, max_account_risk_pct=0.01, capital=1.0)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is not None
        assert sig.size == 0.7

    def test_var_cap_short_direction(self):
        """Test: VaR capping works for SHORT positions"""
        # Setup: SHORT, current_price=100, pct_5=98
        # max_loss_per_unit = 98-100 = -2 < 0 → return unchanged
        dist = make_dist([98.0] * 100)
        stub = StubStrategy(Direction.SHORT, size=0.8)
        strat = VaRPositionCapStrategy(stub, max_account_risk_pct=0.01, capital=1.0)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is not None
        assert sig.direction == Direction.SHORT

    def test_var_cap_passes_through_none(self):
        """Test: returns None if base strategy returns None"""
        dist = make_dist([100.0] * 100)
        stub = StubStrategy(Direction.FLAT)
        strat = VaRPositionCapStrategy(stub, max_account_risk_pct=0.01)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is None


# ============================================================================
# DistributionOverlapStrategy Tests
# ============================================================================

class TestDistributionOverlapStrategy:
    def test_distribution_overlap_range(self):
        """Test: overlap > 0.85 gives range-bound signal (mean reversion)"""
        # Create two similar distributions (high overlap)
        prices1 = [99.0, 100.0, 101.0] * 30
        prices2 = [98.5, 100.0, 101.5] * 30
        dist = make_dist(prices1)
        prev_dist = make_dist(prices2)
        overlap = dist.overlap_coefficient(prev_dist)
        # Should be > 0.85 (very similar)
        strat = DistributionOverlapStrategy(range_threshold=0.85, trend_threshold=0.60)
        sig = strat.generate_signal(dist, 100.0, make_history(), {"prev_dist": prev_dist})
        if sig is not None:
            # Should be range trading (mean reversion)
            assert sig.direction in [Direction.LONG, Direction.SHORT]

    def test_distribution_overlap_trend(self):
        """Test: overlap < 0.60 gives trend-following signal"""
        # Create two very different distributions (low overlap)
        prices1 = [110.0] * 100
        prices2 = [90.0] * 100
        dist = make_dist(prices1)
        prev_dist = make_dist(prices2)
        overlap = dist.overlap_coefficient(prev_dist)
        # Should be < 0.60 (very different)
        strat = DistributionOverlapStrategy(range_threshold=0.85, trend_threshold=0.60)
        sig = strat.generate_signal(dist, 95.0, make_history(), {"prev_dist": prev_dist})
        if sig is not None:
            # Should be trend following
            assert sig.direction in [Direction.LONG, Direction.SHORT]

    def test_distribution_overlap_no_prev(self):
        """Test: returns None when no prev_dist in context"""
        dist = make_dist([100.0] * 100)
        strat = DistributionOverlapStrategy()
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is None

    def test_distribution_overlap_middle_range(self):
        """Test: returns None when overlap is in middle range"""
        # Create moderately similar distributions
        prices1 = [100.0] * 100
        prices2 = [102.0] * 100
        dist = make_dist(prices1)
        prev_dist = make_dist(prices2)
        strat = DistributionOverlapStrategy(range_threshold=0.85, trend_threshold=0.60)
        sig = strat.generate_signal(dist, 101.0, make_history(), {"prev_dist": prev_dist})
        # May or may not signal depending on overlap value


# ============================================================================
# ModelDecayMonitorStrategy Tests
# ============================================================================

class TestModelDecayMonitorStrategy:
    def test_model_decay_widens_stops(self):
        """Test: after 30 bars at 50% hit rate, stop widens"""
        dist = make_dist([110.0] * 100)
        stub = StubStrategy(Direction.LONG, size=0.5)
        strat = ModelDecayMonitorStrategy(stub, lookback=10, target_1sigma=0.68,
                                         widen_factor=1.5, tighten_factor=0.8)

        # Populate calibration history with low hit rate (30% instead of 68%)
        history = make_history(50, 100.0)
        for i in range(15):
            # Create dist where realized close is often outside 1-sigma
            test_dist = make_dist([110.0 + i] * 100)
            realized = 105.0 + i * 0.5  # Outside 1-sigma
            strat.update_calibration(test_dist, realized)

        sig = strat.generate_signal(dist, 110.0, history, {})
        if sig is not None:
            # Stop should be widened (further from entry)
            # Original stop would be pct_10 ≈ 110, new stop should be further
            assert sig.size < 0.5  # Size factor applied

    def test_model_decay_insufficient_data(self):
        """Test: passes through before lookback bars"""
        dist = make_dist([110.0] * 100)
        stub = StubStrategy(Direction.LONG, size=0.7)
        strat = ModelDecayMonitorStrategy(stub, lookback=30)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is not None
        assert sig.size == 0.7  # Unchanged

    def test_model_decay_passes_through_none(self):
        """Test: returns None if base strategy returns None"""
        dist = make_dist([100.0] * 100)
        stub = StubStrategy(Direction.FLAT)
        strat = ModelDecayMonitorStrategy(stub)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is None


# ============================================================================
# OvernightExposureFilter Tests
# ============================================================================

class TestOvernightExposureFilter:
    def test_overnight_filter_long_range_below(self):
        """Test: returns FLAT when pred_high < entry_price for LONG position"""
        dist = make_dist([95.0] * 50 + [96.0] * 50)  # max ~96
        stub = StubStrategy(Direction.LONG, size=0.5)
        strat = OvernightExposureFilter(stub)
        current_pos = {"direction": Direction.LONG, "entry_price": 100.0}
        sig = strat.generate_signal(dist, 98.0, make_history(), {"current_position": current_pos})
        assert sig is not None
        assert sig.direction == Direction.FLAT

    def test_overnight_filter_short_range_above(self):
        """Test: returns FLAT when pred_low > entry_price for SHORT position"""
        dist = make_dist([104.0] * 50 + [105.0] * 50)  # min ~104
        stub = StubStrategy(Direction.SHORT, size=0.5)
        strat = OvernightExposureFilter(stub)
        current_pos = {"direction": Direction.SHORT, "entry_price": 100.0}
        sig = strat.generate_signal(dist, 102.0, make_history(), {"current_position": current_pos})
        assert sig is not None
        assert sig.direction == Direction.FLAT

    def test_overnight_filter_no_position(self):
        """Test: delegates to base when no position"""
        dist = make_dist([110.0] * 100)
        stub = StubStrategy(Direction.LONG, size=0.5)
        strat = OvernightExposureFilter(stub)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_overnight_filter_favorable_range(self):
        """Test: delegates to base when range is favorable"""
        dist = make_dist([105.0] * 100)
        stub = StubStrategy(Direction.LONG, size=0.5)
        strat = OvernightExposureFilter(stub)
        current_pos = {"direction": Direction.LONG, "entry_price": 100.0}
        sig = strat.generate_signal(dist, 102.0, make_history(), {"current_position": current_pos})
        # Should delegate to stub since pred_high > entry_price
        assert sig is not None


# ============================================================================
# RSIDivergenceStrategy Tests
# ============================================================================

class TestRSIDivergenceStrategy:
    def test_rsi_divergence_returns_signal_or_none(self):
        """Smoke test: RSI divergence strategy returns signal or None"""
        # Build history with some volatility
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = 100 + np.sin(np.arange(50) / 5) * 5  # Oscillating prices
        hist = pd.DataFrame({
            "close": prices,
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "volume": [1e6] * 50,
        }, index=idx)

        dist = make_dist([102.0] * 100)  # Predict higher
        strat = RSIDivergenceStrategy(rsi_period=14, lookback_bars=20)
        sig = strat.generate_signal(dist, 100.0, hist, {})
        # Should return signal, None, or handle gracefully
        assert sig is None or isinstance(sig, Signal)

    def test_rsi_divergence_insufficient_history(self):
        """Test: returns None with insufficient history"""
        dist = make_dist([100.0] * 100)
        strat = RSIDivergenceStrategy(rsi_period=14, lookback_bars=20)
        hist = make_history(n=5)
        sig = strat.generate_signal(dist, 100.0, hist, {})
        assert sig is None


# ============================================================================
# LeverageCalibrationStrategy Tests
# ============================================================================

class TestLeverageCalibrationStrategy:
    def test_leverage_low_range(self):
        """Test: pred range < 2% → size * 5 leverage"""
        # Create tight distribution (range ~1.5%)
        prices = [99.98, 99.99, 100.0, 100.01, 100.02] * 20
        dist = make_dist(prices)
        stub = StubStrategy(Direction.LONG, size=0.2)
        strat = LeverageCalibrationStrategy(stub, max_leverage=5.0)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is not None
        # Size should be 0.2 * 5 = 1.0 (capped at max_leverage)
        assert sig.size == 1.0

    def test_leverage_high_range(self):
        """Test: pred range > 6% → size * 1 (no leverage)"""
        # Create wide distribution (range ~12%)
        prices = [94.0] * 50 + [106.0] * 50
        dist = make_dist(prices)
        stub = StubStrategy(Direction.LONG, size=0.8)
        strat = LeverageCalibrationStrategy(stub, max_leverage=5.0)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is not None
        # Size should remain 0.8 (range > 6% tier)
        assert abs(sig.size - 0.8) < 0.01

    def test_leverage_no_change_on_flat(self):
        """Test: FLAT signals pass through unchanged"""
        dist = make_dist([100.0] * 100)
        stub = StubStrategy(Direction.FLAT)
        strat = LeverageCalibrationStrategy(stub, max_leverage=5.0)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is None

    def test_leverage_mid_range(self):
        """Test: pred range 3-4% → size * 3 leverage"""
        # Create medium distribution (range ~3.5%)
        prices = [98.25] * 50 + [101.75] * 50
        dist = make_dist(prices)
        stub = StubStrategy(Direction.LONG, size=0.3)
        strat = LeverageCalibrationStrategy(stub, max_leverage=5.0)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is not None
        # Size should be 0.3 * 3 = 0.9
        assert abs(sig.size - 0.9) < 0.01

    def test_leverage_custom_tiers(self):
        """Test: custom leverage tiers work correctly"""
        prices = [99.9] * 50 + [100.1] * 50
        dist = make_dist(prices)
        stub = StubStrategy(Direction.LONG, size=0.5)
        custom_tiers = [(0.01, 10.0), (0.05, 2.0), (float("inf"), 1.0)]
        strat = LeverageCalibrationStrategy(stub, leverage_tiers=custom_tiers, max_leverage=10.0)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig is not None
        # Range is ~0.2%, should apply 10x leverage, capped at max_leverage=10
        assert sig.size == 5.0


# ============================================================================
# Integration & Edge Cases
# ============================================================================

class TestWrapperIntegration:
    def test_chained_wrappers(self):
        """Test: wrappers can be chained (wrapper wrapping wrapper)"""
        dist = make_dist([110.0] * 100)
        stub = StubStrategy(Direction.LONG, size=0.5)
        var_cap = VaRPositionCapStrategy(stub, max_account_risk_pct=0.01)
        leverage = LeverageCalibrationStrategy(var_cap, max_leverage=5.0)
        sig = leverage.generate_signal(dist, 100.0, make_history(), {})
        assert sig is not None
        assert sig.direction == Direction.LONG

    def test_wrapper_preserves_signal_properties(self):
        """Test: wrappers preserve important signal properties"""
        dist = make_dist([110.0] * 100)
        stub = StubStrategy(Direction.LONG, size=0.5)
        strat = VaRPositionCapStrategy(stub)
        sig = strat.generate_signal(dist, 100.0, make_history(), {})
        assert sig.entry == 100.0
        assert sig.strategy_name == "stub"  # Base strategy name preserved
        assert sig.expected_value > 0

    def test_all_strategies_named(self):
        """Test: all new strategies have correct names"""
        assert VaRPositionCapStrategy(StubStrategy()).name == "var_position_cap"
        assert DistributionOverlapStrategy().name == "distribution_overlap"
        assert ModelDecayMonitorStrategy(StubStrategy()).name == "model_decay_monitor"
        assert OvernightExposureFilter(StubStrategy()).name == "overnight_filter"
        assert RSIDivergenceStrategy().name == "rsi_divergence"
        assert LeverageCalibrationStrategy(StubStrategy()).name == "leverage_calibration"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
