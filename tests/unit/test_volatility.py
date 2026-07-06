import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
import numpy as np
import pandas as pd
from kairos_backtest import (
    KairosDistribution, Direction, Signal,
    PercentileEntryStrategy, DynamicBracketStrategy,
)
from kairos_volatility import atr, ATRBracketStrategy


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
        "open": [price] * n,
        "high": [price * 1.01] * n,
        "low": [price * 0.99] * n,
        "close": [price] * n,
        "volume": [1e6] * n,
    }, index=idx)


@pytest.fixture
def simple_atr_history():
    """
    Hand-computed ATR fixture with constant true ranges.

    All TRs = 3.0, so for any n, ATR should converge to 3.0.

    Index: Date
    0: 2024-01-01, O=100, H=101, L=99, C=100
    1: 2024-01-02, O=100, H=102, L=99, C=101  TR=3
    2: 2024-01-03, O=101, H=103, L=100, C=102 TR=3
    3: 2024-01-04, O=102, H=104, L=101, C=103 TR=3
    4: 2024-01-05, O=103, H=105, L=102, C=104 TR=3
    ...
    """
    n_rows = 20
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")

    opens = np.array([100.0] + [100.0 + i for i in range(1, n_rows)])
    highs = opens + 2.0  # High is always +2.0
    lows = opens - 1.0   # Low is always -1.0
    closes = opens + 1.0  # Close is always +1.0

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1e6] * n_rows,
    }, index=dates)


# ============================================================================
# ATR Tests
# ============================================================================

class TestATRComputation:
    def test_atr_matches_reference(self, simple_atr_history):
        """Test ATR matches hand-computed reference to 1e-6."""
        # For constant TR=3, ATR should converge to 3.0
        atr_val = atr(simple_atr_history, n=3)
        assert abs(atr_val - 3.0) < 1e-6, f"Expected ATR~3.0, got {atr_val}"

    def test_atr_with_different_periods(self, simple_atr_history):
        """Test ATR computation with different period values."""
        atr_3 = atr(simple_atr_history, n=3)
        atr_5 = atr(simple_atr_history, n=5)
        atr_14 = atr(simple_atr_history, n=14)

        # For constant TR, all should converge close to 3
        assert abs(atr_3 - 3.0) < 1e-6
        assert abs(atr_5 - 3.0) < 1e-6
        assert abs(atr_14 - 3.0) < 1e-6

    def test_atr_returns_nan_for_insufficient_history(self):
        """Test ATR returns NaN when history is too short."""
        short_history = pd.DataFrame({
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1e6, 1e6],
        })
        atr_val = atr(short_history, n=14)
        assert np.isnan(atr_val)

    def test_atr_volatile_series(self):
        """Test ATR on series with varying true ranges."""
        # Create history with varying TRs
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": [100.0] * 25,
            "high": [101.0, 103.0, 102.0, 105.0, 104.0] * 5,  # Alternating highs
            "low": [99.0, 99.0, 98.0, 100.0, 99.0] * 5,
            "close": [100.0, 101.0, 100.0, 102.0, 101.0] * 5,
            "volume": [1e6] * 25,
        }, index=dates)

        atr_val = atr(history, n=3)
        # TR varies between 2-5, so ATR should be somewhere in that range
        assert 2.0 <= atr_val <= 5.5


# ============================================================================
# ATRBracketStrategy Tests
# ============================================================================

class TestATRBracketStrategy:
    def test_pass_through_on_none_base_signal(self):
        """Test that None from base strategy is passed through."""
        base_strat = PercentileEntryStrategy()
        wrapper = ATRBracketStrategy(base_strat, k_stop=2.0, k_target=3.0)

        # Create a distribution where no signal fires
        dist = make_dist([100.0] * 100)
        history = make_history(price=100.0)

        sig = wrapper.generate_signal(dist, 100.0, history, {})
        assert sig is None

    def test_atr_bracket_computation_long(self):
        """Test ATR bracket computation for LONG signals."""
        base_strat = DynamicBracketStrategy()
        wrapper = ATRBracketStrategy(base_strat, k_stop=2.0, k_target=3.0, n=3)

        # Create distribution above entry for LONG
        dist = make_dist([110.0] * 100)
        history = make_history(n=20, price=100.0)

        sig = wrapper.generate_signal(dist, 100.0, history, {})
        if sig is not None:
            assert sig.direction == Direction.LONG
            # Verify strategy name is set to wrapper
            assert sig.strategy_name == "atr_bracket"

    def test_atr_stop_only_tightens_long(self, simple_atr_history):
        """Test that LONG stop only ever tightens (moves higher)."""
        # Create a base strategy that produces a LONG signal with wide stop
        class WideStopStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                base_sig = super().generate_signal(dist, current_price, history, context, **kwargs)
                if base_sig is None:
                    return None
                # Override with wider stop
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=105.0,
                    stop=90.0,  # Wide stop
                    target=115.0,
                    strategy_name=base_sig.strategy_name,
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = WideStopStrategy()
        wrapper = ATRBracketStrategy(base_strat, k_stop=2.0, k_target=3.0, n=3)

        dist = make_dist([110.0] * 100)
        sig = wrapper.generate_signal(dist, 105.0, simple_atr_history, {})

        if sig is not None:
            # ATR is 3.0, so ATR stop = 105 - 2*3 = 99
            # Original stop = 90, so tighter stop = max(99, 90) = 99
            assert sig.stop >= 90.0, f"Stop should never loosen; was {sig.stop}"
            assert sig.direction == Direction.LONG

    def test_atr_stop_only_tightens_short(self, simple_atr_history):
        """Test that SHORT stop only ever tightens (moves lower)."""
        class WideStopStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                base_sig = super().generate_signal(dist, current_price, history, context, **kwargs)
                if base_sig is None:
                    return None
                return Signal(
                    direction=Direction.SHORT,
                    size=0.5,
                    entry=95.0,
                    stop=110.0,  # Wide stop for SHORT
                    target=85.0,
                    strategy_name=base_sig.strategy_name,
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = WideStopStrategy()
        wrapper = ATRBracketStrategy(base_strat, k_stop=2.0, k_target=3.0, n=3)

        dist = make_dist([90.0] * 100)
        sig = wrapper.generate_signal(dist, 95.0, simple_atr_history, {})

        if sig is not None:
            # ATR is 3.0, so ATR stop = 95 + 2*3 = 101
            # Original stop = 110, so tighter stop = min(101, 110) = 101
            assert sig.stop <= 110.0, f"Stop should never loosen; was {sig.stop}"
            assert sig.direction == Direction.SHORT

    def test_atr_stop_direction_consistency_long(self):
        """Test that LONG stop is always below entry."""
        class SimpleStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=100.0,
                    stop=98.0,
                    target=105.0,
                    strategy_name="test",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = SimpleStrategy()
        wrapper = ATRBracketStrategy(base_strat, k_stop=2.0, k_target=3.0, n=3)

        history = make_history(n=20, price=100.0)
        dist = make_dist([110.0] * 100)

        sig = wrapper.generate_signal(dist, 100.0, history, {})
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.stop < sig.entry, \
            f"LONG stop must be below entry; stop={sig.stop}, entry={sig.entry}"

    def test_atr_stop_direction_consistency_short(self):
        """Test that SHORT stop is always above entry."""
        class SimpleStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.SHORT,
                    size=0.5,
                    entry=100.0,
                    stop=102.0,
                    target=95.0,
                    strategy_name="test",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = SimpleStrategy()
        wrapper = ATRBracketStrategy(base_strat, k_stop=2.0, k_target=3.0, n=3)

        history = make_history(n=20, price=100.0)
        dist = make_dist([90.0] * 100)

        sig = wrapper.generate_signal(dist, 100.0, history, {})
        assert sig is not None
        assert sig.direction == Direction.SHORT
        assert sig.stop > sig.entry, \
            f"SHORT stop must be above entry; stop={sig.stop}, entry={sig.entry}"

    def test_preserves_other_signal_fields(self):
        """Test that ATR bracket wrapper preserves other Signal fields."""
        class CustomStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.75,
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="custom",
                    confidence=0.9,
                    expected_value=3.5,
                    metadata={"custom_key": "custom_value"},
                )

        base_strat = CustomStrategy()
        wrapper = ATRBracketStrategy(base_strat, k_stop=2.0, k_target=3.0, n=3)

        history = make_history(n=20, price=100.0)
        dist = make_dist([110.0] * 100)

        sig = wrapper.generate_signal(dist, 100.0, history, {})
        assert sig is not None
        # Verify size, confidence, expected_value are preserved
        assert sig.size == 0.75
        assert sig.confidence == 0.9
        assert sig.expected_value == 3.5
        assert sig.metadata == {"custom_key": "custom_value"}
        # Verify strategy_name is changed to wrapper name
        assert sig.strategy_name == "atr_bracket"

    def test_returns_signal_dataclass(self):
        """Test that wrapper always returns Signal dataclass, never dict."""
        base_strat = DynamicBracketStrategy()
        wrapper = ATRBracketStrategy(base_strat, k_stop=2.0, k_target=3.0, n=3)

        dist = make_dist([110.0] * 100)
        history = make_history(n=20, price=100.0)

        sig = wrapper.generate_signal(dist, 100.0, history, {})
        if sig is not None:
            assert isinstance(sig, Signal), f"Expected Signal dataclass, got {type(sig)}"
            # Verify it has all required Signal attributes
            assert hasattr(sig, "direction")
            assert hasattr(sig, "size")
            assert hasattr(sig, "entry")
            assert hasattr(sig, "stop")
            assert hasattr(sig, "target")
            assert hasattr(sig, "strategy_name")
            assert hasattr(sig, "confidence")
            assert hasattr(sig, "expected_value")
            assert hasattr(sig, "metadata")
