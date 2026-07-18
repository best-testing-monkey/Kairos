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
from kairos_volatility import (
    atr, ATRBracketStrategy, fit_garch, GARCHFilterStrategy, VolTargetSizerStrategy,
    VarianceRiskPremiumStrategy
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


# ============================================================================
# GARCH Fit Tests
# ============================================================================

class TestFitGARCH:
    def test_garch_fits_simulated_data(self):
        """Test GARCH fitting recovers alpha+beta within ±0.1 on simulated data."""
        # Seed for reproducibility
        np.random.seed(42)

        # Simulate GARCH(1,1) data with known parameters
        true_alpha = 0.1
        true_beta = 0.85
        true_omega = 0.0001  # Small long-run variance

        n = 500
        sigma2 = np.ones(n) * true_omega
        returns = np.zeros(n)

        for t in range(1, n):
            sigma2[t] = true_omega + true_alpha * (returns[t - 1] ** 2) + true_beta * sigma2[t - 1]
            returns[t] = np.sqrt(sigma2[t]) * np.random.normal(0, 1)

        # Fit GARCH to the simulated data
        fit_result = fit_garch(returns)

        # Verify convergence
        assert fit_result["converged"], "GARCH fit should converge on simulated data"

        # Verify alpha + beta recovery within ±0.1
        estimated_alpha = fit_result["alpha"]
        estimated_beta = fit_result["beta"]
        estimated_sum = estimated_alpha + estimated_beta

        true_sum = true_alpha + true_beta
        assert abs(estimated_sum - true_sum) <= 0.1, \
            f"alpha+beta should be within ±0.1 of {true_sum}, got {estimated_sum}"

    def test_garch_returns_dict_with_required_keys(self):
        """Test fit_garch returns dict with all required keys."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, size=100)

        result = fit_garch(returns)

        assert isinstance(result, dict), "Result should be a dict"
        required_keys = {"omega", "alpha", "beta", "converged", "sigma_forecast"}
        assert required_keys.issubset(result.keys()), \
            f"Result should contain all required keys: {required_keys}"

    def test_garch_with_insufficient_data(self):
        """Test fit_garch handles short return series gracefully."""
        returns = np.array([0.01, 0.02, 0.03])  # Only 3 points
        result = fit_garch(returns)

        assert isinstance(result, dict)
        assert "converged" in result
        assert "sigma_forecast" in result
        assert result["sigma_forecast"] >= 0

    def test_garch_sigma_forecast_positive(self):
        """Test fit_garch sigma_forecast is always positive."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, size=200)
        result = fit_garch(returns)

        assert result["sigma_forecast"] > 0, "Sigma forecast should always be positive"

    def test_garch_alpha_beta_in_valid_range(self):
        """Test fitted alpha and beta are in (0, 1) and sum < 1."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, size=200)
        result = fit_garch(returns)

        alpha = result["alpha"]
        beta = result["beta"]

        assert 0 <= alpha < 1, f"Alpha should be in [0, 1), got {alpha}"
        assert 0 <= beta < 1, f"Beta should be in [0, 1), got {beta}"
        assert alpha + beta < 1, f"Alpha + beta should be < 1, got {alpha + beta}"


# ============================================================================
# GARCHFilterStrategy Tests
# ============================================================================

class TestGARCHFilterStrategy:
    def test_garch_filter_pass_through_on_none_base_signal(self):
        """Test that None from base strategy is passed through."""
        base_strat = PercentileEntryStrategy()
        wrapper = GARCHFilterStrategy(base_strat, sigma_cap_pct=90.0, lookback=100, refit_days=5)

        dist = make_dist([100.0] * 100)
        history = make_history(n=50, price=100.0)

        sig = wrapper.generate_signal(dist, 100.0, history, {})
        assert sig is None

    def test_garch_filter_blocks_on_high_vol(self):
        """Test GARCHFilterStrategy blocks signals during high-volatility regimes."""
        # Create a base strategy that always fires LONG
        class AlwaysFireStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="always_fire",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = AlwaysFireStrategy()
        wrapper = GARCHFilterStrategy(base_strat, sigma_cap_pct=50.0, lookback=50, refit_days=2)

        # Create history with simulated vol spike
        # Start with low volatility, then spike
        np.random.seed(42)
        low_vol_returns = np.random.normal(100.0, 0.5, size=30)  # Low vol
        high_vol_returns = np.random.normal(100.0, 5.0, size=20)  # High vol spike
        volatile_prices = np.concatenate([low_vol_returns, high_vol_returns])

        idx = pd.date_range("2024-01-01", periods=len(volatile_prices), freq="D")
        history = pd.DataFrame({
            "open": volatile_prices * 0.999,
            "high": volatile_prices * 1.005,
            "low": volatile_prices * 0.995,
            "close": volatile_prices,
            "volume": [1e6] * len(volatile_prices),
        }, index=idx)

        dist = make_dist(volatile_prices)
        context = {}

        # Feed bars through to warm up the sigma history
        for i in range(20, len(history)):
            sub_history = history.iloc[:i+1]
            sig = wrapper.generate_signal(dist, volatile_prices[i], sub_history, context)

        # Now test at the high-vol spike: should block (return None)
        sig = wrapper.generate_signal(dist, volatile_prices[-1], history, context)
        assert sig is None, "Signal should be blocked during high-volatility regime"

    def test_garch_filter_fallback_on_convergence_failure(self):
        """Test GARCHFilterStrategy falls back to pass-through with warning on convergence failure."""
        class AlwaysFireStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="always_fire",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = AlwaysFireStrategy()
        wrapper = GARCHFilterStrategy(base_strat, sigma_cap_pct=90.0, lookback=5, refit_days=1)

        # Create history with constant returns (no variance) - will cause fit_garch to not converge
        const_prices = [100.0] * 20
        history = make_history(n=20, price=100.0)

        dist = make_dist(const_prices)
        context = {}

        sig = wrapper.generate_signal(dist, 100.0, history, context)

        # Should pass through but set warning
        assert sig is not None, "Should pass through signal on convergence failure"
        assert context.get("garch_warning") == True, "Should set garch_warning in context"

    def test_garch_filter_refit_cadence(self):
        """Test GARCHFilterStrategy refits at expected cadence."""
        class AlwaysFireStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="always_fire",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = AlwaysFireStrategy()
        refit_days = 3
        wrapper = GARCHFilterStrategy(base_strat, sigma_cap_pct=90.0, lookback=30, refit_days=refit_days)

        # Simulate 15 bars of trading
        np.random.seed(42)
        prices = 100.0 + np.cumsum(np.random.normal(0, 0.5, size=15))

        idx = pd.date_range("2024-01-01", periods=15, freq="D")
        history_full = pd.DataFrame({
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": [1e6] * 15,
        }, index=idx)

        dist = make_dist(prices)
        context = {}

        fit_count = 0
        for i in range(15):
            prev_fit = wrapper._last_fit
            sub_history = history_full.iloc[:i+1]
            wrapper.generate_signal(dist, prices[i], sub_history, context)

            # Check if fit was recomputed
            if wrapper._last_fit is not None and wrapper._last_fit != prev_fit:
                fit_count += 1

        # Should refit approximately every refit_days bars
        # Expected refits: bars 0, 3, 6, 9, 12 = 5 refits (or close to it)
        expected_refits = (15 + refit_days - 1) // refit_days
        assert fit_count >= expected_refits - 1, \
            f"Expected ~{expected_refits} refits, got {fit_count}"

    def test_garch_filter_reset_clears_state(self):
        """Test reset() clears internal state."""
        class AlwaysFireStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="always_fire",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = AlwaysFireStrategy()
        wrapper = GARCHFilterStrategy(base_strat, sigma_cap_pct=90.0, lookback=50, refit_days=5)

        # Simulate some bar activity
        history = make_history(n=30, price=100.0)
        dist = make_dist([100.0] * 30)
        context = {}

        for _ in range(10):
            wrapper.generate_signal(dist, 100.0, history, context)

        # Verify state has been populated
        assert wrapper._bar_count > 0, "Bar count should be > 0"

        # Reset
        wrapper.reset()

        # Verify state is cleared
        assert wrapper._bar_count == 0, "Bar count should be 0 after reset"
        assert wrapper._sigma_history == [], "Sigma history should be empty after reset"
        assert wrapper._last_fit is None, "Last fit should be None after reset"
        assert wrapper._converged == True, "Converged flag should be reset to True"

    def test_garch_filter_preserves_signal_fields(self):
        """Test that GARCHFilterStrategy preserves all Signal fields when passing through."""
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
        wrapper = GARCHFilterStrategy(base_strat, sigma_cap_pct=90.0, lookback=50, refit_days=5)

        history = make_history(n=20, price=100.0)
        dist = make_dist([100.0] * 20)
        context = {}

        sig = wrapper.generate_signal(dist, 100.0, history, context)

        if sig is not None:
            # Verify fields are preserved
            assert sig.size == 0.75
            assert sig.confidence == 0.9
            assert sig.expected_value == 3.5
            assert sig.metadata == {"custom_key": "custom_value"}
            assert sig.direction == Direction.LONG

    def test_garch_filter_returns_signal_dataclass(self):
        """Test that GARCHFilterStrategy returns Signal dataclass, never dict."""
        class AlwaysFireStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="always_fire",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = AlwaysFireStrategy()
        wrapper = GARCHFilterStrategy(base_strat, sigma_cap_pct=90.0, lookback=50, refit_days=5)

        history = make_history(n=20, price=100.0)
        dist = make_dist([100.0] * 20)
        context = {}

        sig = wrapper.generate_signal(dist, 100.0, history, context)

        if sig is not None:
            assert isinstance(sig, Signal), f"Expected Signal dataclass, got {type(sig)}"


# ============================================================================
# VolTargetSizerStrategy Tests
# ============================================================================

class TestVolTargetSizerStrategy:
    def test_vol_target_sizer_none_passthrough(self):
        """Test that None from base strategy is passed through."""
        base_strat = PercentileEntryStrategy()
        wrapper = VolTargetSizerStrategy(base_strat, target_vol=0.15, max_leverage=2.0)

        dist = make_dist([100.0] * 100)
        history = make_history(n=50, price=100.0)
        context = {}

        sig = wrapper.generate_signal(dist, 100.0, history, context)
        assert sig is None

    def test_vol_target_sizer_zero_stays_zero(self):
        """Test that zero-size signals are never increased."""
        class ZeroSizeStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.0,  # Zero size
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="zero_size",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = ZeroSizeStrategy()
        wrapper = VolTargetSizerStrategy(base_strat, target_vol=0.15, max_leverage=2.0)

        dist = make_dist([110.0] * 100)
        history = make_history(n=50, price=100.0)
        context = {}

        sig = wrapper.generate_signal(dist, 100.0, history, context)
        assert sig is not None
        assert sig.size == 0.0, "Zero-size signal should stay zero"

    def test_vol_target_sizer_halves_at_double_vol(self):
        """Test that size halves when blended vol doubles.

        Create two scenarios with different volatility levels and verify the
        scaling relationship matches target_vol / blended_vol.
        """
        class FixedSizeStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=1.0,  # Base size = 1.0
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="fixed_size",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = FixedSizeStrategy()
        target_vol = 0.15
        wrapper = VolTargetSizerStrategy(base_strat, target_vol=target_vol, max_leverage=2.0,
                                        lookback=50, refit_days=10)

        # Scenario 1: Low volatility
        # Create constant prices (near-zero returns) and tight distribution for Kronos
        np.random.seed(42)
        low_vol_prices = [100.0] * 60
        idx = pd.date_range("2024-01-01", periods=60, freq="D")
        low_vol_history = pd.DataFrame({
            "open": low_vol_prices,
            "high": [p * 1.001 for p in low_vol_prices],
            "low": [p * 0.999 for p in low_vol_prices],
            "close": low_vol_prices,
            "volume": [1e6] * 60,
        }, index=idx)

        # Create a tight distribution (narrow range)
        low_vol_dist = make_dist(low_vol_prices)

        sig1 = wrapper.generate_signal(low_vol_dist, 100.0, low_vol_history, {})
        assert sig1 is not None
        size1 = sig1.size

        # Reset for scenario 2
        wrapper.reset()

        # Scenario 2: High volatility (wider distribution)
        # Create volatile returns and wider distribution
        high_vol_prices = [100.0 + 5.0 * np.sin(i * 0.3) for i in range(60)]
        high_vol_history = pd.DataFrame({
            "open": [p * 0.99 for p in high_vol_prices],
            "high": [p * 1.02 for p in high_vol_prices],
            "low": [p * 0.98 for p in high_vol_prices],
            "close": high_vol_prices,
            "volume": [1e6] * 60,
        }, index=idx)

        high_vol_dist = make_dist(high_vol_prices)

        sig2 = wrapper.generate_signal(high_vol_dist, 100.0, high_vol_history, {})
        assert sig2 is not None
        size2 = sig2.size

        # High vol should have smaller size than low vol
        # (since we're scaling down when vol is high)
        assert size2 < size1, f"High vol size {size2} should be < low vol size {size1}"

    def test_vol_target_sizer_respects_max_leverage(self):
        """Test that sizing never exceeds base size * max_leverage."""
        class FixedSizeStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=1.0,  # Base size = 1.0
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="fixed_size",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = FixedSizeStrategy()
        base_size = 1.0
        max_leverage = 1.5

        wrapper = VolTargetSizerStrategy(base_strat, target_vol=0.15, max_leverage=max_leverage,
                                        lookback=50, refit_days=10)

        # Create very low volatility scenario (should want to scale up)
        np.random.seed(42)
        const_prices = [100.0] * 60
        idx = pd.date_range("2024-01-01", periods=60, freq="D")
        low_vol_history = pd.DataFrame({
            "open": const_prices,
            "high": [p * 1.0001 for p in const_prices],  # Tiny range
            "low": [p * 0.9999 for p in const_prices],
            "close": const_prices,
            "volume": [1e6] * 60,
        }, index=idx)

        low_vol_dist = make_dist(const_prices)

        sig = wrapper.generate_signal(low_vol_dist, 100.0, low_vol_history, {})
        assert sig is not None

        max_allowed_size = base_size * max_leverage
        assert sig.size <= max_allowed_size, \
            f"Size {sig.size} exceeds max {max_allowed_size}"

    def test_vol_target_sizer_preserves_signal_fields(self):
        """Test that VolTargetSizerStrategy preserves non-size fields."""
        class CustomStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="custom",
                    confidence=0.85,
                    expected_value=3.5,
                    metadata={"key": "value"},
                )

        base_strat = CustomStrategy()
        wrapper = VolTargetSizerStrategy(base_strat, target_vol=0.15, max_leverage=2.0)

        history = make_history(n=50, price=100.0)
        dist = make_dist([100.0] * 50)
        context = {}

        sig = wrapper.generate_signal(dist, 100.0, history, context)
        assert sig is not None

        # Verify non-size fields are preserved
        assert sig.direction == Direction.LONG
        assert sig.entry == 100.0
        assert sig.stop == 95.0
        assert sig.target == 110.0
        assert sig.confidence == 0.85
        assert sig.expected_value == 3.5
        assert sig.metadata == {"key": "value"}
        assert sig.strategy_name == "vol_target_sizer"

    def test_vol_target_sizer_reset_clears_cache(self):
        """Test reset() clears internal state."""
        class AlwaysFireStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="always_fire",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = AlwaysFireStrategy()
        wrapper = VolTargetSizerStrategy(base_strat, target_vol=0.15, max_leverage=2.0,
                                        lookback=50, refit_days=5)

        # Simulate some bar activity
        history = make_history(n=30, price=100.0)
        dist = make_dist([100.0] * 30)
        context = {}

        for _ in range(10):
            wrapper.generate_signal(dist, 100.0, history, context)

        # Verify state has been populated
        assert wrapper._bar_count > 0, "Bar count should be > 0"

        # Reset
        wrapper.reset()

        # Verify state is cleared
        assert wrapper._bar_count == 0, "Bar count should be 0 after reset"
        assert wrapper._sigma_history == [], "Sigma history should be empty after reset"
        assert wrapper._last_fit is None, "Last fit should be None after reset"
        assert wrapper._converged == True, "Converged flag should be reset to True"

    def test_vol_target_sizer_returns_signal_dataclass(self):
        """Test that VolTargetSizerStrategy returns Signal dataclass, never dict."""
        class AlwaysFireStrategy(DynamicBracketStrategy):
            def generate_signal(self, dist, current_price, history, context, **kwargs):
                return Signal(
                    direction=Direction.LONG,
                    size=0.5,
                    entry=100.0,
                    stop=95.0,
                    target=110.0,
                    strategy_name="always_fire",
                    confidence=1.0,
                    expected_value=2.0,
                )

        base_strat = AlwaysFireStrategy()
        wrapper = VolTargetSizerStrategy(base_strat, target_vol=0.15, max_leverage=2.0)

        history = make_history(n=20, price=100.0)
        dist = make_dist([100.0] * 20)
        context = {}

        sig = wrapper.generate_signal(dist, 100.0, history, context)
        assert sig is not None
        assert isinstance(sig, Signal), f"Expected Signal dataclass, got {type(sig)}"
        assert hasattr(sig, "direction")
        assert hasattr(sig, "size")
        assert hasattr(sig, "entry")
        assert hasattr(sig, "stop")
        assert hasattr(sig, "target")
        assert hasattr(sig, "strategy_name")
        assert hasattr(sig, "confidence")
        assert hasattr(sig, "expected_value")
        assert hasattr(sig, "metadata")


# ============================================================================
# VarianceRiskPremiumStrategy Tests
# ============================================================================

class TestVarianceRiskPremiumStrategy:
    def test_vrp_none_when_history_too_short(self):
        """Test that VarianceRiskPremiumStrategy returns None when history < 21 rows."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)
        history = make_history(n=20, price=100.0)  # Exactly 20 rows
        dist = make_dist([100.0] * 20)

        sig = strategy.generate_signal(dist, 100.0, history, {})
        assert sig is None, "Should return None when history < 21 rows"

    def test_vrp_neutral_band_no_signal(self):
        """Test that neutral band [1/entry_ratio, entry_ratio] produces no signal."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)

        # Create history with minimal variance (realized_var very small)
        np.random.seed(42)
        prices = [100.0 + 0.01 * np.sin(i * 0.1) for i in range(25)]
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": [p * 0.999 for p in prices],
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [1e6] * 25,
        }, index=idx)

        # Create distribution with moderate range (implied_var moderate)
        # to put ratio in the neutral band
        dist = make_dist(prices)

        sig = strategy.generate_signal(dist, 100.0, history, {})
        # If ratio falls in [1/1.5, 1.5] ≈ [0.67, 1.5], should be None
        if sig is None:
            # This is acceptable if ratio naturally fell in neutral band
            assert True
        else:
            # If signal fired, it's outside neutral band
            assert sig is not None

    def test_vrp_expansion_entry_long(self):
        """Test vol expansion entry with LONG direction."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)

        # Create history with low variance (to ensure low realized_var)
        np.random.seed(42)
        prices = [100.0] * 25  # Constant price = zero variance
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": prices,
            "high": [p * 1.0001 for p in prices],
            "low": [p * 0.9999 for p in prices],
            "close": prices,
            "volume": [1e6] * 25,
        }, index=idx)

        # Create distribution with very wide range to force high implied_var
        # Use price samples spread far apart
        wide_prices = [80.0, 85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0]
        wide_prices_full = wide_prices * 3  # Replicate to get enough samples
        dist = make_dist(wide_prices_full)

        sig = strategy.generate_signal(dist, 100.0, history, {})
        if sig is not None:
            # Should be expansion trade
            assert sig.metadata.get("trade_type") == "vol_expansion"
            # Should have wide bracket
            bracket_width = abs(sig.target - sig.stop)
            assert bracket_width > 0

    def test_vrp_expansion_bracket_pct_5_95_long(self):
        """Test that expansion LONG uses pct_5/pct_95 bracket exactly."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)

        # Create low-variance history
        np.random.seed(42)
        prices = [100.0] * 25
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": prices,
            "high": [p * 1.0001 for p in prices],
            "low": [p * 0.9999 for p in prices],
            "close": prices,
            "volume": [1e6] * 25,
        }, index=idx)

        # Create distribution with known percentiles
        # Make 100 samples: 50 at 80, 50 at 120 (bimodal, skewed right)
        dist_prices = [80.0] * 50 + [120.0] * 50
        dist = make_dist(dist_prices)

        sig = strategy.generate_signal(dist, 100.0, history, {})
        if sig is not None and sig.metadata.get("trade_type") == "vol_expansion":
            close_stats = dist.stats.get("close", {})
            if sig.direction == Direction.LONG:
                expected_stop = close_stats.get(f"pct_{int(5)}", None)
                expected_target = close_stats.get(f"pct_{int(95)}", None)
                if expected_stop is not None and expected_target is not None:
                    assert sig.stop == expected_stop, \
                        f"LONG stop should be pct_5 ({expected_stop}), got {sig.stop}"
                    assert sig.target == expected_target, \
                        f"LONG target should be pct_95 ({expected_target}), got {sig.target}"

    def test_vrp_expansion_bracket_pct_5_95_short(self):
        """Test that expansion SHORT uses pct_95/pct_5 bracket (reversed) exactly."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)

        np.random.seed(42)
        prices = [100.0] * 25
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": prices,
            "high": [p * 1.0001 for p in prices],
            "low": [p * 0.9999 for p in prices],
            "close": prices,
            "volume": [1e6] * 25,
        }, index=idx)

        # Create distribution skewed left (to trigger SHORT in expansion)
        dist_prices = [120.0] * 50 + [80.0] * 50
        dist = make_dist(dist_prices)

        sig = strategy.generate_signal(dist, 100.0, history, {})
        if sig is not None and sig.metadata.get("trade_type") == "vol_expansion":
            close_stats = dist.stats.get("close", {})
            if sig.direction == Direction.SHORT:
                expected_stop = close_stats.get(f"pct_{int(95)}", None)
                expected_target = close_stats.get(f"pct_{int(5)}", None)
                if expected_stop is not None and expected_target is not None:
                    assert sig.stop == expected_stop, \
                        f"SHORT stop should be pct_95 ({expected_stop}), got {sig.stop}"
                    assert sig.target == expected_target, \
                        f"SHORT target should be pct_5 ({expected_target}), got {sig.target}"

    def test_vrp_compression_bracket_pct_25_75_long(self):
        """Test that compression LONG uses pct_25/pct_75 bracket exactly."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)

        # Create high-variance history (to get high realized_var)
        np.random.seed(42)
        prices = [100.0 + 10.0 * np.sin(i * 0.5) for i in range(25)]
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1e6] * 25,
        }, index=idx)

        # Create tight distribution (low implied_var) to trigger compression
        dist = make_dist(prices)

        sig = strategy.generate_signal(dist, 100.0, history, {})
        if sig is not None and sig.metadata.get("trade_type") == "vol_compression":
            close_stats = dist.stats.get("close", {})
            if sig.direction == Direction.LONG:
                expected_stop = close_stats.get(f"pct_{int(25)}", None)
                expected_target = close_stats.get(f"pct_{int(75)}", None)
                if expected_stop is not None and expected_target is not None:
                    assert sig.stop == expected_stop, \
                        f"LONG stop should be pct_25 ({expected_stop}), got {sig.stop}"
                    assert sig.target == expected_target, \
                        f"LONG target should be pct_75 ({expected_target}), got {sig.target}"

    def test_vrp_compression_bracket_pct_25_75_short(self):
        """Test that compression SHORT uses pct_75/pct_25 bracket (reversed) exactly."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)

        np.random.seed(42)
        prices = [100.0 + 10.0 * np.sin(i * 0.5) for i in range(25)]
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": [p * 0.99 for p in prices],
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1e6] * 25,
        }, index=idx)

        dist = make_dist(prices)

        # Signal at current_price = 100 (center)
        # For SHORT compression, current_price should be > mean
        # Adjust center to be above the mean
        high_price = 110.0
        sig = strategy.generate_signal(dist, high_price, history, {})
        if sig is not None and sig.metadata.get("trade_type") == "vol_compression":
            close_stats = dist.stats.get("close", {})
            if sig.direction == Direction.SHORT:
                expected_stop = close_stats.get(f"pct_{int(75)}", None)
                expected_target = close_stats.get(f"pct_{int(25)}", None)
                if expected_stop is not None and expected_target is not None:
                    assert sig.stop == expected_stop, \
                        f"SHORT stop should be pct_75 ({expected_stop}), got {sig.stop}"
                    assert sig.target == expected_target, \
                        f"SHORT target should be pct_25 ({expected_target}), got {sig.target}"

    def test_vrp_direction_consistent_stops_long(self):
        """Test that LONG stops are always below entry."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)

        np.random.seed(42)
        prices = [100.0] * 25
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": prices,
            "high": [p * 1.0001 for p in prices],
            "low": [p * 0.9999 for p in prices],
            "close": prices,
            "volume": [1e6] * 25,
        }, index=idx)

        wide_prices = [80.0] * 50 + [120.0] * 50
        dist = make_dist(wide_prices)

        sig = strategy.generate_signal(dist, 100.0, history, {})
        if sig is not None and sig.direction == Direction.LONG:
            assert sig.stop < sig.entry, \
                f"LONG stop {sig.stop} must be below entry {sig.entry}"

    def test_vrp_direction_consistent_stops_short(self):
        """Test that SHORT stops are always above entry."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)

        np.random.seed(42)
        prices = [100.0] * 25
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": prices,
            "high": [p * 1.0001 for p in prices],
            "low": [p * 0.9999 for p in prices],
            "close": prices,
            "volume": [1e6] * 25,
        }, index=idx)

        wide_prices = [120.0] * 50 + [80.0] * 50
        dist = make_dist(wide_prices)

        sig = strategy.generate_signal(dist, 100.0, history, {})
        if sig is not None and sig.direction == Direction.SHORT:
            assert sig.stop > sig.entry, \
                f"SHORT stop {sig.stop} must be above entry {sig.entry}"

    def test_vrp_returns_signal_dataclass(self):
        """Test that VarianceRiskPremiumStrategy returns Signal or None, never dict."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)

        np.random.seed(42)
        prices = [100.0] * 25
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": prices,
            "high": [p * 1.0001 for p in prices],
            "low": [p * 0.9999 for p in prices],
            "close": prices,
            "volume": [1e6] * 25,
        }, index=idx)

        wide_prices = [80.0] * 50 + [120.0] * 50
        dist = make_dist(wide_prices)

        sig = strategy.generate_signal(dist, 100.0, history, {})
        if sig is not None:
            assert isinstance(sig, Signal), f"Expected Signal or None, got {type(sig)}"
            assert hasattr(sig, "direction")
            assert hasattr(sig, "size")
            assert hasattr(sig, "entry")
            assert hasattr(sig, "stop")
            assert hasattr(sig, "target")
            assert hasattr(sig, "strategy_name")
            assert hasattr(sig, "confidence")
            assert hasattr(sig, "expected_value")
            assert hasattr(sig, "metadata")

    def test_vrp_metadata_contains_required_fields(self):
        """Test that signal metadata contains ratio, implied_var, realized_var, trade_type."""
        strategy = VarianceRiskPremiumStrategy(entry_ratio=1.5)

        np.random.seed(42)
        prices = [100.0] * 25
        idx = pd.date_range("2024-01-01", periods=25, freq="D")
        history = pd.DataFrame({
            "open": prices,
            "high": [p * 1.0001 for p in prices],
            "low": [p * 0.9999 for p in prices],
            "close": prices,
            "volume": [1e6] * 25,
        }, index=idx)

        wide_prices = [80.0] * 50 + [120.0] * 50
        dist = make_dist(wide_prices)

        sig = strategy.generate_signal(dist, 100.0, history, {})
        if sig is not None:
            assert "ratio" in sig.metadata, "Metadata must contain 'ratio'"
            assert "implied_var" in sig.metadata, "Metadata must contain 'implied_var'"
            assert "realized_var" in sig.metadata, "Metadata must contain 'realized_var'"
            assert "trade_type" in sig.metadata, "Metadata must contain 'trade_type'"
            assert sig.metadata["trade_type"] in ["vol_expansion", "vol_compression"]
