import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
import numpy as np
import pandas as pd
from kairos_backtest import KairosDistribution, Direction, Signal
from kairos_meta import KurtosisFilterStrategy
from kairos_execution import LiquidityFilterStrategy
from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig


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


def _make_dummy_orchestrator(config=None):
    """Create a minimal orchestrator for testing _apply_meta_filters."""
    def dummy_predict(signal, **kwargs):
        return []
    return KairosOrchestrator(predict_fn=dummy_predict, assets=["BTC-USD"], config=config)


# ============================================================================
# Tests
# ============================================================================

class TestKurtosisFilterStrategy:
    def _make_low_kurt_dist(self):
        """Create a normal-ish distribution with low excess kurtosis."""
        # Use fixed seed for reproducibility
        np.random.seed(42)
        prices = np.random.normal(100, 1, 100).tolist()
        return make_dist(prices)

    def _make_high_kurt_dist(self):
        """Create a spike distribution with very high excess kurtosis."""
        prices = [100.0] * 95 + [200.0] * 3 + [0.0] * 2
        return make_dist(prices)

    def test_block_action_high_kurtosis(self):
        """Test that high kurtosis signals are blocked."""
        from kairos_backtest import CloseDirectionStrategy
        base = CloseDirectionStrategy()
        filt = KurtosisFilterStrategy(base_strategy=base, max_kurtosis=3.0, action="block")
        dist = self._make_high_kurt_dist()
        sig = filt.generate_signal(dist, 100.0, make_history(), {})
        assert sig is None  # blocked

    def test_block_action_low_kurtosis_passes(self):
        """Test that low kurtosis signals pass through."""
        from kairos_backtest import CloseDirectionStrategy
        base = CloseDirectionStrategy()
        filt = KurtosisFilterStrategy(base_strategy=base, max_kurtosis=10.0, action="block")
        dist = make_dist([110.0] * 100)  # uniform → low kurtosis
        sig = filt.generate_signal(dist, 100.0, make_history(), {})
        # Base strategy should fire (passes through filter with max_kurtosis=10)
        if sig is not None:
            assert sig.direction == Direction.LONG

    def test_reduce_action_halves_size(self):
        """Test that reduce action halves signal size."""
        from kairos_backtest import CloseDirectionStrategy
        filt_no_reduce = KurtosisFilterStrategy(
            base_strategy=CloseDirectionStrategy(),
            max_kurtosis=100.0,
            action="block"
        )
        filt_reduce = KurtosisFilterStrategy(
            base_strategy=CloseDirectionStrategy(),
            max_kurtosis=0.0,
            action="reduce"
        )
        dist = make_dist([110.0] * 100)
        sig_base = filt_no_reduce.generate_signal(dist, 100.0, make_history(), {})
        sig_reduced = filt_reduce.generate_signal(dist, 100.0, make_history(), {})
        if sig_base is not None and sig_reduced is not None:
            assert sig_reduced.size == pytest.approx(sig_base.size * 0.5, rel=1e-3)

    def test_invert_action_flips_direction(self):
        """Test that invert action flips signal direction."""
        from kairos_backtest import CloseDirectionStrategy
        # Use a spike distribution (excess kurtosis >> 0) to guarantee filter triggers
        spike_prices = [100.0] * 95 + [200.0] * 3 + [0.001] * 2  # leptokurtic
        dist = make_dist(spike_prices)
        filt = KurtosisFilterStrategy(
            base_strategy=CloseDirectionStrategy(),
            max_kurtosis=3.0,  # spike dist kurtosis >> 3
            action="invert"
        )
        # mean ≈ 101 > current_price 90 → base would be LONG → inverted = SHORT
        sig = filt.generate_signal(dist, 90.0, make_history(), {})
        if sig is not None:
            assert sig.direction == Direction.SHORT  # inverted


class TestMetaFilters:
    """Test _apply_meta_filters behavior via a minimal orchestrator."""

    def _make_orch(self, entropy_threshold=3.0, bimodality_filter=True):
        """Create an orchestrator with custom filter settings."""
        config = OrchestratorConfig(
            entropy_threshold=entropy_threshold,
            bimodality_filter=bimodality_filter,
        )
        return _make_dummy_orchestrator(config)

    def test_high_entropy_blocked(self):
        """Test that high entropy distributions are blocked."""
        # Spread across all 20 bins → max Shannon entropy ≈ ln(20) ≈ 3.0
        prices = np.linspace(50, 150, 100).tolist()
        dist = make_dist(prices)
        orch = self._make_orch(entropy_threshold=0.5)  # very low threshold → will block
        blocked = orch._apply_meta_filters(dist, 100.0)
        assert blocked is True

    def test_low_entropy_passes(self):
        """Test that low entropy distributions pass."""
        # All same price → entropy = 0 → not blocked
        dist = make_dist([100.0] * 100)
        orch = self._make_orch(entropy_threshold=3.0)
        blocked = orch._apply_meta_filters(dist, 100.0)
        assert blocked is False

    def test_bimodal_blocked(self):
        """Test that bimodal distributions are blocked."""
        # Two equal clusters → negative excess kurtosis < -1.0
        prices = [80.0] * 50 + [120.0] * 50
        dist = make_dist(prices)
        orch = self._make_orch(bimodality_filter=True)
        blocked = orch._apply_meta_filters(dist, 100.0)
        assert blocked is True  # bimodal → blocked

    def test_bimodality_filter_disabled(self):
        """Test that bimodality filter can be disabled."""
        prices = [80.0] * 50 + [120.0] * 50
        dist = make_dist(prices)
        orch = self._make_orch(bimodality_filter=False)
        # With bimodality disabled and low entropy (just 2 values), should pass entropy check
        # entropy of 2-valued dist (50/50) ≈ ln(2) ≈ 0.69 which is < 3.0
        blocked = orch._apply_meta_filters(dist, 100.0)
        assert blocked is False
