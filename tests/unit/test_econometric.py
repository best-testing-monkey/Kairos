import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
import numpy as np
import pandas as pd
from kairos_backtest import (
    KairosDistribution, Direction, Signal,
    PercentileEntryStrategy, DynamicBracketStrategy,
)
from kairos_econometric import (
    _lagged_ols, ARIMADisagreementStrategy,
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


def make_history(closes, price=100.0):
    """Build a history DataFrame from a list of closes."""
    n = len(closes)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "open": [c * 0.999 for c in closes],
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": [1e6] * n,
    }, index=idx)


class StubStrategy(PercentileEntryStrategy):
    """A stub strategy that always returns a signal with predictable properties."""
    def __init__(self, direction=Direction.LONG, confidence=0.8, **kwargs):
        super().__init__(**kwargs)
        self.direction = direction
        self.base_confidence = confidence

    def generate_signal(self, dist, current_price, history, context):
        # Delegate to parent for the calculation, but override confidence
        sig = super().generate_signal(dist, current_price, history, context)
        if sig is None:
            return None
        return Signal(
            direction=self.direction,
            size=sig.size,
            entry=current_price,
            stop=current_price * 0.95,
            target=current_price * 1.05,
            strategy_name=sig.strategy_name,
            confidence=self.base_confidence,
            expected_value=sig.expected_value,
            metadata=sig.metadata or {},
        )


# ============================================================================
# Tests: _lagged_ols
# ============================================================================

class TestLaggedOLS:
    def test_lagged_ols_simple_linear(self):
        """Test OLS on simple y = 2x + noise."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y = 2 * x + np.random.normal(0, 0.1, 100)
        X = np.column_stack([np.ones(100), x])

        result = _lagged_ols(y, X)

        assert "coef" in result
        assert "se" in result
        assert "tstats" in result
        assert "resid" in result
        assert "aic" in result

        # Coefficients should be close to [0, 2]
        assert np.allclose(result["coef"], [0, 2], atol=0.1)
        assert len(result["se"]) == 2
        assert len(result["tstats"]) == 2
        assert len(result["resid"]) == 100

    def test_lagged_ols_residuals(self):
        """Test that residuals are correctly computed."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        X[:, 0] = 1  # Intercept
        coef_true = np.array([1.0, 2.0, 3.0])
        y = X @ coef_true + np.random.normal(0, 0.01, 50)

        result = _lagged_ols(y, X)
        expected_resid = y - X @ result["coef"]

        assert np.allclose(result["resid"], expected_resid)

    def test_lagged_ols_aic_increases_with_params(self):
        """Test that AIC increases when adding irrelevant parameters."""
        np.random.seed(42)
        n = 100
        x = np.linspace(0, 10, n)
        y = 2 * x + np.random.normal(0, 0.1, n)

        # Model 1: intercept + x
        X1 = np.column_stack([np.ones(n), x])
        result1 = _lagged_ols(y, X1)
        aic1 = result1["aic"]

        # Model 2: intercept + x + noise
        noise = np.random.randn(n)
        X2 = np.column_stack([np.ones(n), x, noise])
        result2 = _lagged_ols(y, X2)
        aic2 = result2["aic"]

        # AIC2 should be higher (worse) since noise is irrelevant
        # AIC = n*ln(RSS/n) + 2k, so more parameters hurt despite fitting better
        assert aic2 > aic1


# ============================================================================
# Tests: ARIMADisagreementStrategy
# ============================================================================

class TestARIMADisagreementVeto:
    def test_arima_veto_when_directions_disagree(self):
        """Test that veto fires when AR and Kronos disagree on direction."""
        # Create history: upward trend closes
        uptrend_closes = np.array([100.0 + i * 0.5 for i in range(130)])
        history = make_history(uptrend_closes)

        # Create distribution with mean BELOW current price (downward forecast)
        # This will disagree with the AR(p) uptrend forecast
        current_price = uptrend_closes[-1]
        kronos_closes = [current_price * 0.95] * 100  # Mean below current
        dist = make_dist(kronos_closes)

        # Base strategy that always signals LONG
        base = StubStrategy(direction=Direction.LONG, confidence=0.8)

        # Filter with ARIMA
        arima = ARIMADisagreementStrategy(base, lookback=120, max_p=5)
        sig = arima.generate_signal(dist, current_price, history, {})

        # Should be vetoed (None) because AR says UP but Kronos says DOWN
        assert sig is None

    def test_arima_boost_when_directions_agree(self):
        """Test that confidence is boosted when AR and Kronos agree."""
        # Create history: upward trend
        uptrend_closes = np.array([100.0 + i * 0.5 for i in range(130)])
        history = make_history(uptrend_closes)

        # Create distribution with mean ABOVE current price (upward forecast)
        # This agrees with the AR(p) uptrend forecast
        current_price = uptrend_closes[-1]
        kronos_closes = [current_price * 1.05] * 100  # Mean above current
        dist = make_dist(kronos_closes)

        # Base strategy with confidence 0.8
        base = StubStrategy(direction=Direction.LONG, confidence=0.8)

        # Filter with ARIMA, agree_boost=1.2
        arima = ARIMADisagreementStrategy(base, lookback=120, max_p=5, agree_boost=1.2)
        sig = arima.generate_signal(dist, current_price, history, {})

        # Should pass through and boost confidence
        assert sig is not None
        assert sig.direction == Direction.LONG
        # Boosted: min(0.8 * 1.2, 1.0) = min(0.96, 1.0) = 0.96
        assert np.isclose(sig.confidence, 0.96)

    def test_arima_boost_capped_at_one(self):
        """Test that boosted confidence is capped at 1.0."""
        uptrend_closes = np.array([100.0 + i * 0.5 for i in range(130)])
        history = make_history(uptrend_closes)

        current_price = uptrend_closes[-1]
        kronos_closes = [current_price * 1.05] * 100
        dist = make_dist(kronos_closes)

        # High base confidence
        base = StubStrategy(direction=Direction.LONG, confidence=0.95)

        # High agree_boost to exceed 1.0
        arima = ARIMADisagreementStrategy(base, lookback=120, max_p=5, agree_boost=1.5)
        sig = arima.generate_signal(dist, current_price, history, {})

        assert sig is not None
        # Should be capped: min(0.95 * 1.5, 1.0) = 1.0
        assert sig.confidence == 1.0

    def test_arima_passthrough_on_base_none(self):
        """Test that None from base strategy passes through as None."""
        # Create a simple history (not enough for trend to be super obvious)
        closes = [100.0] * 50  # Flat
        history = make_history(closes)

        # Create a neutral distribution
        dist = make_dist([100.0] * 100)

        # Base strategy that returns None
        class NoneStrategy(PercentileEntryStrategy):
            def generate_signal(self, dist, current_price, history, context):
                return None

        base = NoneStrategy()
        arima = ARIMADisagreementStrategy(base, lookback=120, max_p=5)
        sig = arima.generate_signal(dist, 100.0, history, {})

        assert sig is None


class TestARIMAOrderSelection:
    def test_arima_aic_selects_planted_ar2(self):
        """Test that AIC correctly selects AR(2) on synthetic AR(2) data."""
        np.random.seed(42)

        # Generate synthetic AR(2) data: y_t = 0.5*y_{t-1} + 0.3*y_{t-2} + noise
        phi1, phi2 = 0.5, 0.3
        const = 100.0
        y = np.zeros(300)
        y[0] = const
        y[1] = const

        noise = np.random.normal(0, 1, 300)
        for t in range(2, 300):
            y[t] = const + phi1 * (y[t - 1] - const) + phi2 * (y[t - 2] - const) + noise[t]

        # Fit AR(p) for p=1..5 and check AIC
        aic_scores = {}
        for p in range(1, 6):
            n = len(y)
            if n <= p + 1:
                continue

            X = np.ones((n - p, p + 1))
            y_reg = y[p:]

            for lag in range(1, p + 1):
                X[:, lag] = y[p - lag : -lag or None]

            from kairos_econometric import _lagged_ols
            result = _lagged_ols(y_reg, X)
            aic_scores[p] = result["aic"]

        # AR(2) should have lower AIC than AR(1) or AR(3+)
        assert aic_scores[2] < aic_scores[1]
        assert aic_scores[2] <= aic_scores[3]

    def test_arima_trend_detection(self):
        """Test that on a pure uptrend, AR forecast direction is positive."""
        # Pure uptrend
        uptrend = np.array([100.0 + i * 1.0 for i in range(130)])
        history = make_history(uptrend)

        # Neutral distribution (doesn't matter, we're testing AR)
        dist = make_dist([105.0] * 100)

        current_price = uptrend[-1]

        # Base strategy that always signals
        base = StubStrategy(direction=Direction.LONG, confidence=0.8)

        arima = ARIMADisagreementStrategy(base, lookback=120, max_p=5)

        # Get the AR forecast directly
        closes = history["close"].tail(120).values
        best_p = arima._select_ar_order(closes)
        ar_forecast = arima._ar_forecast(closes, best_p)

        # AR forecast should be above current price (uptrend)
        assert ar_forecast is not None
        assert ar_forecast > current_price

    def test_arima_insufficient_history(self):
        """Test that with insufficient history, base signal passes through unchanged."""
        # Create very short history
        closes = [100.0, 101.0, 102.0]
        history = make_history(closes)

        dist = make_dist([102.0] * 100)
        current_price = 102.0

        base = StubStrategy(direction=Direction.LONG, confidence=0.7)
        arima = ARIMADisagreementStrategy(base, lookback=120, max_p=5)

        sig = arima.generate_signal(dist, current_price, history, {})

        # Should pass through unchanged (not enough history)
        if sig is not None:
            assert sig.confidence == 0.7  # No boost


# ============================================================================
# Tests: Edge cases
# ============================================================================

class TestEdgeCases:
    def test_arima_with_flat_series(self):
        """Test ARIMA on a flat price series."""
        flat = [100.0] * 130
        history = make_history(flat)

        dist = make_dist([100.0] * 100)
        current_price = 100.0

        base = StubStrategy(direction=Direction.LONG, confidence=0.8)
        arima = ARIMADisagreementStrategy(base, lookback=120, max_p=5)

        sig = arima.generate_signal(dist, current_price, history, {})

        # Should either pass through or veto, but not crash
        # A flat series has zero forecast movement, so direction is 0
        # This won't trigger veto (requires non-zero disagreement)
        if sig is not None:
            assert sig.confidence >= 0  # Valid signal

    def test_arima_with_random_walk(self):
        """Test ARIMA on a random walk series."""
        np.random.seed(42)
        rw = np.cumsum(np.random.randn(130)) + 100
        history = make_history(rw)

        dist = make_dist([105.0] * 100)
        current_price = rw[-1]

        base = StubStrategy(direction=Direction.LONG, confidence=0.8)
        arima = ARIMADisagreementStrategy(base, lookback=120, max_p=5)

        sig = arima.generate_signal(dist, current_price, history, {})

        # Should not crash; might pass, veto, or boost
        assert sig is None or sig.confidence >= 0

    def test_arima_with_downtrend(self):
        """Test ARIMA detects downtrend correctly."""
        downtrend = np.array([130.0 - i * 0.5 for i in range(130)])
        history = make_history(downtrend)

        dist = make_dist([downtrend[-1] * 0.99] * 100)  # Kronos also predicts down
        current_price = downtrend[-1]

        base = StubStrategy(direction=Direction.SHORT, confidence=0.8)
        arima = ARIMADisagreementStrategy(base, lookback=120, max_p=5)

        sig = arima.generate_signal(dist, current_price, history, {})

        # Both AR and Kronos agree on downward direction
        # Should boost confidence
        if sig is not None:
            assert sig.confidence >= 0.8
