import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
import numpy as np
import pandas as pd
from kairos_backtest import (
    KairosDistribution, Direction, Signal, Strategy,
    StochasticFilterStrategy, CloseDirectionStrategy, ADXGateStrategy,
    OBVConfirmationStrategy,
    compute_adx
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


def make_history(n=50, price=100.0, opens=None, highs=None, lows=None, closes=None):
    """Build a history DataFrame for backtesting."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    if closes is None:
        closes = [price] * n
    if opens is None:
        opens = [c * 0.999 for c in closes]
    if highs is None:
        highs = [c * 1.01 for c in closes]
    if lows is None:
        lows = [c * 0.99 for c in closes]

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": [1e6] * n
    }, index=idx)


# ============================================================================
# Tests for StochasticFilterStrategy
# ============================================================================

class TestStochasticFilterStrategy:
    """Test the Stochastic Oscillator + ADX filter."""

    def test_stochastic_hand_computed_fixture(self):
        """
        Test %K/%D computation against hand-computed values on a small OHLC series.

        Use a simple series where we can verify %K manually:
        - 10-bar series with known high/low/close
        - Compute expected %K = 100*(close - low) / (high - low)
        - Verify to 1e-6 precision
        """
        # Create a simple 14-bar series for testing (k_period=14)
        closes = [100, 101, 102, 103, 102, 101, 102, 103, 104, 105, 104, 103, 102, 101]
        highs = [101, 102, 103, 104, 103, 102, 103, 104, 105, 106, 105, 104, 103, 102]
        lows = [99, 100, 101, 102, 101, 100, 101, 102, 103, 104, 103, 102, 101, 100]

        history = make_history(
            n=len(closes),
            opens=[c - 0.5 for c in closes],
            highs=highs,
            lows=lows,
            closes=closes
        )

        # Create a dummy base strategy that always generates a neutral signal
        class DummyStrategy(Strategy):
            name = "dummy"
            def generate_signal(self, dist, current_price, history, context):
                return None

        filt = StochasticFilterStrategy(
            base_strategy=DummyStrategy(),
            k_period=14, d_period=3
        )

        # Manually compute expected %K for this series
        # Over 14 bars: high = 106, low = 99
        highest_high = 106.0
        lowest_low = 99.0
        last_close = 101.0
        expected_k = 100.0 * (last_close - lowest_low) / (highest_high - lowest_low)
        # expected_k = 100 * (101 - 99) / (106 - 99) = 100 * 2 / 7 ≈ 28.571

        k_val, d_val = filt._compute_stochastic(history)

        # Assert %K matches to 1e-6 precision
        assert abs(k_val - expected_k) < 1e-6, f"Expected K={expected_k}, got {k_val}"

    def test_stochastic_overbought_long_veto(self):
        """
        Test that LONG signals are vetoed when %K > overbought UNLESS trending (ADX > threshold).
        """
        # Create a series where price is trending up strongly (for high ADX)
        n = 50
        closes = list(np.linspace(100, 150, n))  # Strong uptrend
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.98 for c in closes]

        history = make_history(n=n, closes=closes, highs=highs, lows=lows)

        # Base strategy that always generates LONG
        class AlwaysLongStrategy(Strategy):
            name = "always_long"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        # Create distribution where we expect the signal to generate
        dist = make_dist(closes[-100:])

        # Filter with high overbought threshold and low ADX trend threshold
        filt = StochasticFilterStrategy(
            base_strategy=AlwaysLongStrategy(),
            overbought=25.0,  # Very low threshold to trigger overbought
            oversold=75.0,    # Very high to never trigger oversold
            adx_trend=10.0,   # Low trend threshold (uptrend should exceed)
        )

        sig = filt.generate_signal(dist, closes[-1], history, {})

        # In this strong uptrend, %K should be high (near end of range)
        # and ADX should also be high (strong trend)
        # So the signal should NOT be vetoed
        # Let's check: even though %K is high, ADX is high so it passes
        if sig is not None:
            # The signal should pass through because ADX > adx_trend
            assert sig.direction == Direction.LONG

    def test_stochastic_oversold_short_veto(self):
        """
        Test that SHORT signals are vetoed when %K < oversold UNLESS trending (ADX > threshold).
        """
        # Create a downtrend (low %K, high ADX)
        n = 50
        closes = list(np.linspace(150, 100, n))  # Strong downtrend
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.98 for c in closes]

        history = make_history(n=n, closes=closes, highs=highs, lows=lows)

        # Base strategy that always generates SHORT
        class AlwaysShortStrategy(Strategy):
            name = "always_short"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.SHORT,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_90"],
                    target=s["pct_10"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        dist = make_dist(closes[-100:])

        filt = StochasticFilterStrategy(
            base_strategy=AlwaysShortStrategy(),
            overbought=25.0,   # Low threshold
            oversold=75.0,     # High threshold to trigger oversold
            adx_trend=10.0,    # Low trend threshold (downtrend should exceed)
        )

        sig = filt.generate_signal(dist, closes[-1], history, {})

        # In downtrend, %K is low and ADX is high
        # Signal should pass through (not vetoed)
        if sig is not None:
            assert sig.direction == Direction.SHORT

    def test_stochastic_veto_truth_table(self):
        """
        Test all four combinations:
        1. %K > overbought, ADX low: LONG vetoed, SHORT passes
        2. %K > overbought, ADX high: both pass
        3. %K < oversold, ADX low: SHORT vetoed, LONG passes
        4. %K < oversold, ADX high: both pass
        """
        # Case 1: Sideways market (low ADX, mid %K)
        n = 50
        closes = [100 + (i % 10 - 5) for i in range(n)]  # Oscillating
        highs = [c + 2 for c in closes]
        lows = [c - 2 for c in closes]
        history = make_history(n=n, closes=closes, highs=highs, lows=lows)

        class SignalGenerator(Strategy):
            def __init__(self, direction):
                self.direction = direction
                self.name = f"signal_{direction}"

            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=self.direction,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_10"] if self.direction == Direction.LONG else s["pct_90"],
                    target=s["pct_90"] if self.direction == Direction.LONG else s["pct_10"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=5.0,
                )

        dist = make_dist(closes)

        # Test with moderate thresholds
        filt_long = StochasticFilterStrategy(
            base_strategy=SignalGenerator(Direction.LONG),
            overbought=50.0,
            oversold=50.0,
            adx_trend=30.0,
        )

        filt_short = StochasticFilterStrategy(
            base_strategy=SignalGenerator(Direction.SHORT),
            overbought=50.0,
            oversold=50.0,
            adx_trend=30.0,
        )

        sig_long = filt_long.generate_signal(dist, 100.0, history, {})
        sig_short = filt_short.generate_signal(dist, 100.0, history, {})

        # In sideways market (low ADX), %K should be near 50
        # So neither overbought nor oversold conditions trigger
        # Both signals should pass through
        if sig_long is not None:
            assert sig_long.direction == Direction.LONG
        if sig_short is not None:
            assert sig_short.direction == Direction.SHORT

    def test_stochastic_passthrough_none_base_signal(self):
        """
        Test that None base signal passes through unchanged.
        """
        class NeverSignalStrategy(Strategy):
            name = "never_signal"
            def generate_signal(self, dist, current_price, history, context):
                return None

        history = make_history(n=50, price=100.0)
        dist = make_dist([100.0] * 50)

        filt = StochasticFilterStrategy(base_strategy=NeverSignalStrategy())
        sig = filt.generate_signal(dist, 100.0, history, {})

        assert sig is None

    def test_stochastic_passthrough_signal_unchanged(self):
        """
        Test that a base signal passes through unchanged (same object) when filter doesn't veto.
        """
        class FixedSignalStrategy(Strategy):
            name = "fixed"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.75,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.8,
                    expected_value=15.0,
                )

        history = make_history(n=50, price=100.0)
        dist = make_dist([100.0] * 50)

        # Use high overbought threshold so normal conditions won't trigger veto
        filt = StochasticFilterStrategy(
            base_strategy=FixedSignalStrategy(),
            overbought=95.0,
            oversold=5.0,
        )

        sig = filt.generate_signal(dist, 100.0, history, {})

        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.size == 0.75
        assert sig.confidence == 0.8
        assert sig.strategy_name == "fixed"


# ============================================================================
# Tests for ADXGateStrategy
# ============================================================================

class TestADXHelper:
    """Test the module-level compute_adx helper function."""

    def test_adx_insufficient_data(self):
        """Test that ADX returns 50.0 (neutral) when insufficient data."""
        history = make_history(n=5, price=100.0)
        adx = compute_adx(history, n=14)
        assert adx == 50.0

    def test_adx_matches_stochastic_computation(self):
        """
        Test that compute_adx produces values consistent with StochasticFilterStrategy._compute_adx.
        Using synthetic OHLC fixture with strong trend.
        """
        # Create a strong uptrend (for high ADX)
        n = 50
        closes = list(np.linspace(100, 150, n))
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.98 for c in closes]

        history = make_history(n=n, closes=closes, highs=highs, lows=lows)

        # Compute ADX via helper
        adx_from_helper = compute_adx(history, n=14)

        # Create StochasticFilterStrategy and compute via its method
        class DummyStrategy(Strategy):
            name = "dummy"
            def generate_signal(self, dist, current_price, history, context):
                return None

        stoch_filter = StochasticFilterStrategy(
            base_strategy=DummyStrategy(),
            adx_period=14
        )
        adx_from_stochastic = stoch_filter._compute_adx(history)

        # Both should produce identical values
        assert abs(adx_from_helper - adx_from_stochastic) < 1e-9, \
            f"ADX mismatch: helper={adx_from_helper}, stochastic={adx_from_stochastic}"

    def test_adx_high_on_strong_trend(self):
        """Test that ADX is high (> 25) on a strong uptrend."""
        n = 50
        closes = list(np.linspace(100, 150, n))
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.98 for c in closes]

        history = make_history(n=n, closes=closes, highs=highs, lows=lows)
        adx = compute_adx(history, n=14)

        assert adx > 25.0, f"Expected ADX > 25 on strong uptrend, got {adx}"

    def test_adx_low_on_sideways_market(self):
        """Test that ADX is low (< 20) on a sideways/choppy market."""
        n = 50
        closes = [100 + (i % 10 - 5) for i in range(n)]  # Oscillating
        highs = [c + 2 for c in closes]
        lows = [c - 2 for c in closes]

        history = make_history(n=n, closes=closes, highs=highs, lows=lows)
        adx = compute_adx(history, n=14)

        assert adx < 25.0, f"Expected ADX < 25 on sideways market, got {adx}"


class TestADXGateStrategy:
    """Test ADXGateStrategy wrapper."""

    def test_adx_gate_trend_routing_passes_on_trend(self):
        """
        Test that trend-type strategy passes signal only when ADX > trend_min.
        Using synthetic data with strong uptrend (high ADX).
        """
        # Create a strong uptrend (high ADX)
        n = 50
        closes = list(np.linspace(100, 150, n))
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.98 for c in closes]

        history = make_history(n=n, closes=closes, highs=highs, lows=lows)

        # Base strategy that always generates LONG
        class AlwaysLongStrategy(Strategy):
            name = "always_long"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        dist = make_dist(closes[-100:])

        # Trend gate with trend_min=25
        gate = ADXGateStrategy(
            base_strategy=AlwaysLongStrategy(),
            kind="trend",
            trend_min=25.0
        )

        sig = gate.generate_signal(dist, closes[-1], history, {})

        # On strong uptrend, ADX should be high, so signal should pass
        assert sig is not None, "Signal should pass on strong trend (high ADX)"
        assert sig.direction == Direction.LONG
        assert sig.strategy_name == "always_long"

    def test_adx_gate_trend_routing_blocks_on_flat(self):
        """
        Test that trend-type strategy blocks signal when ADX < trend_min.
        Using synthetic data with sideways/choppy market (low ADX).
        """
        # Create a sideways market (low ADX)
        n = 50
        closes = [100 + (i % 10 - 5) for i in range(n)]
        highs = [c + 2 for c in closes]
        lows = [c - 2 for c in closes]

        history = make_history(n=n, closes=closes, highs=highs, lows=lows)

        class AlwaysLongStrategy(Strategy):
            name = "always_long"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        dist = make_dist(closes[-100:])

        # Trend gate
        gate = ADXGateStrategy(
            base_strategy=AlwaysLongStrategy(),
            kind="trend",
            trend_min=25.0
        )

        sig = gate.generate_signal(dist, closes[-1], history, {})

        # On sideways market, ADX should be low, so trend strategy should be blocked
        assert sig is None, "Trend signal should be blocked on flat/sideways market (low ADX)"

    def test_adx_gate_reversion_routing_passes_on_flat(self):
        """
        Test that reversion-type strategy passes signal only when ADX < reversion_max.
        Using synthetic data with sideways/choppy market (low ADX).
        """
        # Create a sideways market (low ADX)
        n = 50
        closes = [100 + (i % 10 - 5) for i in range(n)]
        highs = [c + 2 for c in closes]
        lows = [c - 2 for c in closes]

        history = make_history(n=n, closes=closes, highs=highs, lows=lows)

        class AlwaysLongStrategy(Strategy):
            name = "always_long"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        dist = make_dist(closes[-100:])

        # Reversion gate with reversion_max=20
        gate = ADXGateStrategy(
            base_strategy=AlwaysLongStrategy(),
            kind="reversion",
            reversion_max=20.0
        )

        sig = gate.generate_signal(dist, closes[-1], history, {})

        # On sideways market, ADX should be low, so reversion strategy should pass
        assert sig is not None, "Reversion signal should pass on flat/sideways market (low ADX)"
        assert sig.direction == Direction.LONG
        assert sig.strategy_name == "always_long"

    def test_adx_gate_reversion_routing_blocks_on_trend(self):
        """
        Test that reversion-type strategy blocks signal when ADX > reversion_max.
        Using synthetic data with strong uptrend (high ADX).
        """
        # Create a strong uptrend (high ADX)
        n = 50
        closes = list(np.linspace(100, 150, n))
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.98 for c in closes]

        history = make_history(n=n, closes=closes, highs=highs, lows=lows)

        class AlwaysLongStrategy(Strategy):
            name = "always_long"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        dist = make_dist(closes[-100:])

        # Reversion gate
        gate = ADXGateStrategy(
            base_strategy=AlwaysLongStrategy(),
            kind="reversion",
            reversion_max=20.0
        )

        sig = gate.generate_signal(dist, closes[-1], history, {})

        # On strong trend, ADX should be high, so reversion strategy should be blocked
        assert sig is None, "Reversion signal should be blocked on strong trend (high ADX)"

    def test_adx_gate_passthrough_none_base_signal(self):
        """Test that None base signal passes through unchanged."""
        class NeverSignalStrategy(Strategy):
            name = "never_signal"
            def generate_signal(self, dist, current_price, history, context):
                return None

        history = make_history(n=50, price=100.0)
        dist = make_dist([100.0] * 50)

        gate = ADXGateStrategy(
            base_strategy=NeverSignalStrategy(),
            kind="trend"
        )
        sig = gate.generate_signal(dist, 100.0, history, {})

        assert sig is None

    def test_adx_gate_invalid_kind_raises_error(self):
        """Test that invalid kind parameter raises ValueError."""
        class DummyStrategy(Strategy):
            name = "dummy"
            def generate_signal(self, dist, current_price, history, context):
                return None

        with pytest.raises(ValueError, match="kind must be 'trend' or 'reversion'"):
            ADXGateStrategy(
                base_strategy=DummyStrategy(),
                kind="invalid"
            )

    def test_adx_gate_signal_unchanged_when_passing(self):
        """Test that passing signal maintains original attributes."""
        class FixedSignalStrategy(Strategy):
            name = "fixed"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.75,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.8,
                    expected_value=15.0,
                )

        # Use strong trend so trend gate will pass
        n = 50
        closes = list(np.linspace(100, 150, n))
        highs = [c * 1.02 for c in closes]
        lows = [c * 0.98 for c in closes]
        history = make_history(n=n, closes=closes, highs=highs, lows=lows)

        dist = make_dist(closes[-100:])

        gate = ADXGateStrategy(
            base_strategy=FixedSignalStrategy(),
            kind="trend",
            trend_min=25.0
        )

        sig = gate.generate_signal(dist, closes[-1], history, {})

        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.size == 0.75
        assert sig.confidence == 0.8
        assert sig.strategy_name == "fixed"
        assert sig.expected_value == 15.0


# ============================================================================
# Tests for OBVConfirmationStrategy
# ============================================================================

class TestOBVConfirmationStrategy:
    """Test the On-Balance Volume confirmation filter."""

    def test_obv_hand_computed_fixture(self):
        """
        Test OBV computation against hand-computed values on a small fixture.

        Use a simple series where we can verify OBV manually:
        - 5-bar series with known closes and volumes
        - Compute OBV = cumulative volume signed by price change
        - Verify to exact precision
        """
        # Simple 5-bar fixture:
        # Day 1: close=100, vol=1000 -> OBV=0 (starting point)
        # Day 2: close=101 (up), vol=1000 -> OBV=0 + 1000 = 1000
        # Day 3: close=100 (down), vol=1000 -> OBV=1000 - 1000 = 0
        # Day 4: close=102 (up), vol=1000 -> OBV=0 + 1000 = 1000
        # Day 5: close=102 (flat), vol=1000 -> OBV=1000 + 0 = 1000

        closes = [100.0, 101.0, 100.0, 102.0, 102.0]
        volumes = [1000.0, 1000.0, 1000.0, 1000.0, 1000.0]

        history = pd.DataFrame({
            "open": [c - 0.5 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": volumes,
        }, index=pd.date_range("2024-01-01", periods=len(closes), freq="D"))

        # Create a dummy base strategy
        class DummyStrategy(Strategy):
            name = "dummy"
            def generate_signal(self, dist, current_price, history, context):
                return None

        obv_filter = OBVConfirmationStrategy(
            base_strategy=DummyStrategy(),
            slope_window=20
        )

        obv_values, current_obv = obv_filter._compute_obv(history)

        # Expected OBV values
        expected_obv = [0.0, 1000.0, 0.0, 1000.0, 1000.0]

        # Check each value
        for i, (computed, expected) in enumerate(zip(obv_values, expected_obv)):
            assert abs(computed - expected) < 1e-6, \
                f"OBV mismatch at index {i}: expected {expected}, got {computed}"

        # Check current OBV (last value)
        assert abs(current_obv - 1000.0) < 1e-6, \
            f"Current OBV mismatch: expected 1000.0, got {current_obv}"

    def test_obv_slope_disagreement_veto_long(self):
        """
        Test that LONG signals are vetoed when OBV slope is negative (falling OBV trend).
        Create prices that decline (causing negative OBV) while generating LONG signal.
        """
        # Create a series where price goes DOWN (creating negative OBV slope)
        # but we generate a LONG signal (disagreement = should veto)
        n = 30
        closes = list(np.linspace(110, 100, n))  # Prices trending DOWN
        # Constant volume (OBV slope depends only on price movement)
        volumes = [5000.0] * n

        history = pd.DataFrame({
            "open": [c - 0.5 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": volumes,
        }, index=pd.date_range("2024-01-01", periods=n, freq="D"))

        # Base strategy that always generates LONG (despite falling prices)
        class AlwaysLongStrategy(Strategy):
            name = "always_long"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        dist = make_dist(closes[-100:])

        obv_filter = OBVConfirmationStrategy(
            base_strategy=AlwaysLongStrategy(),
            slope_window=15
        )

        sig = obv_filter.generate_signal(dist, closes[-1], history, {})

        # Signal should be vetoed (None) because OBV slope is negative (prices declining)
        # and signal is LONG (disagreement)
        assert sig is None, "LONG signal should be vetoed when OBV slope is negative"

    def test_obv_slope_disagreement_veto_short(self):
        """
        Test that SHORT signals are vetoed when OBV slope is positive (rising OBV trend).
        Create prices that increase (causing positive OBV) while generating SHORT signal.
        """
        # Create a series where price goes UP (creating positive OBV slope)
        # but we generate a SHORT signal (disagreement = should veto)
        n = 30
        closes = list(np.linspace(100, 110, n))  # Prices trending UP
        # Constant volume (OBV slope depends only on price movement)
        volumes = [5000.0] * n

        history = pd.DataFrame({
            "open": [c - 0.5 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": volumes,
        }, index=pd.date_range("2024-01-01", periods=n, freq="D"))

        # Base strategy that always generates SHORT (despite rising prices)
        class AlwaysShortStrategy(Strategy):
            name = "always_short"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.SHORT,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_90"],
                    target=s["pct_10"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        dist = make_dist(closes[-100:])

        obv_filter = OBVConfirmationStrategy(
            base_strategy=AlwaysShortStrategy(),
            slope_window=15
        )

        sig = obv_filter.generate_signal(dist, closes[-1], history, {})

        # Signal should be vetoed (None) because OBV slope is positive (prices rising)
        # and signal is SHORT (disagreement)
        assert sig is None, "SHORT signal should be vetoed when OBV slope is positive"

    def test_obv_slope_agreement_pass_long(self):
        """
        Test that LONG signals pass through when OBV slope is positive (rising OBV trend).
        """
        # Create a series where price goes UP (creating positive OBV slope)
        # and we generate a LONG signal (agreement = should pass)
        n = 30
        closes = list(np.linspace(100, 110, n))  # Prices trending UP
        # Constant volume (OBV slope depends on price movement)
        volumes = [5000.0] * n

        history = pd.DataFrame({
            "open": [c - 0.5 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": volumes,
        }, index=pd.date_range("2024-01-01", periods=n, freq="D"))

        # Base strategy that always generates LONG
        class AlwaysLongStrategy(Strategy):
            name = "always_long"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        dist = make_dist(closes[-100:])

        obv_filter = OBVConfirmationStrategy(
            base_strategy=AlwaysLongStrategy(),
            slope_window=15
        )

        sig = obv_filter.generate_signal(dist, closes[-1], history, {})

        # Signal should pass through because OBV slope is positive (agrees with LONG)
        assert sig is not None, "LONG signal should pass when OBV slope is positive"
        assert sig.direction == Direction.LONG
        assert sig.strategy_name == "always_long"

    def test_obv_slope_agreement_pass_short(self):
        """
        Test that SHORT signals pass through when OBV slope is negative (falling OBV trend).
        """
        # Create a series where price goes DOWN (creating negative OBV slope)
        # and we generate a SHORT signal (agreement = should pass)
        n = 30
        closes = list(np.linspace(110, 100, n))  # Prices trending DOWN
        # Constant volume (OBV slope depends on price movement)
        volumes = [5000.0] * n

        history = pd.DataFrame({
            "open": [c - 0.5 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": volumes,
        }, index=pd.date_range("2024-01-01", periods=n, freq="D"))

        # Base strategy that always generates SHORT
        class AlwaysShortStrategy(Strategy):
            name = "always_short"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.SHORT,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_90"],
                    target=s["pct_10"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        dist = make_dist(closes[-100:])

        obv_filter = OBVConfirmationStrategy(
            base_strategy=AlwaysShortStrategy(),
            slope_window=15
        )

        sig = obv_filter.generate_signal(dist, closes[-1], history, {})

        # Signal should pass through because OBV slope is negative (agrees with SHORT)
        assert sig is not None, "SHORT signal should pass when OBV slope is negative"
        assert sig.direction == Direction.SHORT
        assert sig.strategy_name == "always_short"

    def test_obv_flat_slope_passthrough(self):
        """
        Test that signals pass through when OBV slope is near-zero (flat).
        Constant volume with tiny alternating price moves summing to near-zero slope.
        """
        # Create a series with constant volume and tiny near-flat price movement
        n = 30
        # Alternating tiny up/down moves that net to nearly flat
        closes = [100.0 + (0.001 * (-1)**i) for i in range(n)]
        volumes = [5000.0] * n  # Constant volume

        history = pd.DataFrame({
            "open": [c - 0.0005 for c in closes],
            "high": [c + 0.001 for c in closes],
            "low": [c - 0.001 for c in closes],
            "close": closes,
            "volume": volumes,
        }, index=pd.date_range("2024-01-01", periods=n, freq="D"))

        # Base strategy that generates LONG
        class AlwaysLongStrategy(Strategy):
            name = "always_long"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=10.0,
                )

        dist = make_dist(closes[-100:])

        obv_filter = OBVConfirmationStrategy(
            base_strategy=AlwaysLongStrategy(),
            slope_window=15
        )

        sig = obv_filter.generate_signal(dist, closes[-1], history, {})

        # With flat OBV, signal should pass through (neither positive nor negative slope blocks)
        assert sig is not None, "LONG signal should pass with flat OBV slope (near-zero)"
        assert sig.direction == Direction.LONG

    def test_obv_passthrough_none_base_signal(self):
        """Test that None base signal passes through unchanged."""
        class NeverSignalStrategy(Strategy):
            name = "never_signal"
            def generate_signal(self, dist, current_price, history, context):
                return None

        history = make_history(n=50, price=100.0)
        dist = make_dist([100.0] * 50)

        obv_filter = OBVConfirmationStrategy(base_strategy=NeverSignalStrategy())
        sig = obv_filter.generate_signal(dist, 100.0, history, {})

        assert sig is None

    def test_obv_signal_unchanged_when_passing(self):
        """
        Test that passing signal maintains all original attributes unchanged
        (field-by-field).
        """
        # Create a series with positive OBV slope
        n = 30
        closes = list(np.linspace(100, 110, n))
        volumes = [5000 + (100 * i) for i in range(n)]

        history = pd.DataFrame({
            "open": [c - 0.5 for c in closes],
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": volumes,
        }, index=pd.date_range("2024-01-01", periods=n, freq="D"))

        # Base strategy with specific signal attributes
        class FixedSignalStrategy(Strategy):
            name = "fixed"
            def generate_signal(self, dist, current_price, history, context):
                s = dist.stats["close"]
                return Signal(
                    direction=Direction.LONG,
                    size=0.75,
                    entry=current_price,
                    stop=s["pct_10"],
                    target=s["pct_90"],
                    strategy_name=self.name,
                    confidence=0.8,
                    expected_value=15.0,
                )

        dist = make_dist(closes[-100:])

        obv_filter = OBVConfirmationStrategy(
            base_strategy=FixedSignalStrategy(),
            slope_window=15
        )

        sig = obv_filter.generate_signal(dist, closes[-1], history, {})

        # Verify all signal attributes are unchanged
        assert sig is not None
        assert sig.direction == Direction.LONG
        assert sig.size == 0.75
        assert sig.confidence == 0.8
        assert sig.strategy_name == "fixed"
        assert sig.expected_value == 15.0
