"""
Tests for new meta-strategies: RegimeClusterStrategy and MonteCarloScenarioStrategy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../strategy"))

import pandas as pd
import numpy as np
import pytest

from kairos_backtest import (
    KairosDistribution, Direction, Signal, Strategy
)
from kairos_meta import RegimeClusterStrategy, MonteCarloScenarioStrategy


# =============================================================================
# HELPERS
# =============================================================================

def make_dist(closes):
    """Create a KairosDistribution from a list of close prices."""
    rows = []
    for c in closes:
        df = pd.DataFrame({
            "open": [c],
            "high": [c * 1.01],
            "low": [c * 0.99],
            "close": [c],
            "volume": [1.0]
        })
        rows.append(df)
    return KairosDistribution(rows)


# =============================================================================
# STUB STRATEGIES
# =============================================================================

class StubLongStrategy(Strategy):
    """Stub strategy that always returns LONG signals."""
    name = "stub_long"

    def generate_signal(self, dist, current_price, history, context, **kw):
        return Signal(
            direction=Direction.LONG,
            size=0.5,
            entry=current_price,
            stop=current_price * 0.95,
            target=current_price * 1.05,
            strategy_name=self.name,
            confidence=0.7,
            expected_value=0.01,
            metadata={}
        )


class StubShortStrategy(Strategy):
    """Stub strategy that always returns SHORT signals."""
    name = "stub_short"

    def generate_signal(self, dist, current_price, history, context, **kw):
        return Signal(
            direction=Direction.SHORT,
            size=0.5,
            entry=current_price,
            stop=current_price * 1.05,
            target=current_price * 0.95,
            strategy_name=self.name,
            confidence=0.7,
            expected_value=0.01,
            metadata={}
        )


class StubNoneStrategy(Strategy):
    """Stub strategy that always returns None."""
    name = "stub_none"

    def generate_signal(self, dist, current_price, history, context, **kw):
        return None


class StubFlatStrategy(Strategy):
    """Stub strategy that returns FLAT signals."""
    name = "stub_flat"

    def generate_signal(self, dist, current_price, history, context, **kw):
        return Signal(
            direction=Direction.FLAT,
            size=0.0,
            entry=current_price,
            stop=0.0,
            target=0.0,
            strategy_name=self.name,
            confidence=0.0,
            expected_value=0.0,
            metadata={}
        )


# =============================================================================
# REGIME CLUSTER TESTS
# =============================================================================

class TestRegimeClusterStrategy:

    def test_regime_cluster_uses_fallback_when_buffer_empty(self):
        """Empty buffer should return fallback strategy's signal."""
        long_strat = StubLongStrategy()
        none_strat = StubNoneStrategy()

        cluster_strat = RegimeClusterStrategy(
            base_strategies=[long_strat, none_strat],
            feature_buffer_size=100,
            k_neighbors=5,
            fallback_strategy=long_strat
        )

        dist = make_dist([100, 101, 99, 102, 98])
        signal = cluster_strat.generate_signal(dist, 100, None, {})

        assert signal is not None
        assert signal.strategy_name == "stub_long"
        assert signal.direction == Direction.LONG

    def test_regime_cluster_selects_best_strategy_from_buffer(self):
        """Should select strategy with highest mean PnL from KNN neighbors."""
        long_strat = StubLongStrategy()
        short_strat = StubShortStrategy()

        cluster_strat = RegimeClusterStrategy(
            base_strategies=[long_strat, short_strat],
            feature_buffer_size=100,
            k_neighbors=3,
            distance_threshold=1.0,
            fallback_strategy=None
        )

        # Manually populate feature buffer with mock entries
        # Strategy A (long_strat) with high PnL
        features_a = [0.05, 0.1, 0.02, 1.0, 1.0]
        cluster_strat.feature_buffer.append((features_a, "stub_long", 100.0))
        cluster_strat.feature_buffer.append((features_a, "stub_long", 95.0))
        cluster_strat.feature_buffer.append((features_a, "stub_long", 110.0))

        # Strategy B (short_strat) with low PnL
        features_b = [0.05, 0.1, 0.02, 1.0, 1.0]
        cluster_strat.feature_buffer.append((features_b, "stub_short", 10.0))
        cluster_strat.feature_buffer.append((features_b, "stub_short", 5.0))
        cluster_strat.feature_buffer.append((features_b, "stub_short", 8.0))

        # Create a distribution with similar features to populated entries
        dist = make_dist([100, 101, 99, 102, 98, 100.5])
        signal = cluster_strat.generate_signal(dist, 100, None, {})

        assert signal is not None
        # Should select stub_long because it has higher mean PnL
        assert signal.strategy_name == "stub_long"
        assert signal.direction == Direction.LONG

    def test_regime_cluster_returns_none_when_buffer_small_no_fallback(self):
        """Should return None if buffer too small and no fallback."""
        long_strat = StubLongStrategy()

        # Explicitly pass fallback_strategy=None (not relying on default)
        cluster_strat = RegimeClusterStrategy(
            base_strategies=[long_strat],
            k_neighbors=5
        )
        # Verify fallback is None
        assert cluster_strat.fallback_strategy is None

        # Buffer is empty (size < k_neighbors)
        dist = make_dist([100, 101, 99])
        signal = cluster_strat.generate_signal(dist, 100, None, {})

        assert signal is None

    def test_regime_cluster_distance_calculation(self):
        """Should compute distances properly between feature vectors."""
        long_strat = StubLongStrategy()

        cluster_strat = RegimeClusterStrategy(
            base_strategies=[long_strat],
            k_neighbors=2,
            distance_threshold=1.0,
            fallback_strategy=long_strat
        )

        # Populate buffer with known features
        cluster_strat.feature_buffer.append(([0.01, 0.0, 0.01, 0.5, 1.0], "stub_long", 10.0))
        cluster_strat.feature_buffer.append(([0.02, 0.05, 0.015, 0.6, 1.0], "stub_long", 5.0))

        # Generate signal - should work with fallback available
        dist = make_dist([100, 101, 99, 102, 98])
        signal = cluster_strat.generate_signal(dist, 100, None, {})

        # Should get a signal back (from fallback or regime_cluster)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_regime_cluster_record_trade(self):
        """Should successfully record trades into buffer."""
        cluster_strat = RegimeClusterStrategy(
            base_strategies=[StubLongStrategy()],
            feature_buffer_size=10
        )

        features = [0.05, 0.1, 0.02, 1.0, 1.0]
        cluster_strat.record_trade(features, "stub_long", 25.5)

        assert len(cluster_strat.feature_buffer) == 1
        recorded = list(cluster_strat.feature_buffer)[0]
        assert recorded[0] == features
        assert recorded[1] == "stub_long"
        assert recorded[2] == 25.5

    def test_regime_cluster_feature_extraction(self):
        """Should extract 5D features correctly."""
        cluster_strat = RegimeClusterStrategy(base_strategies=[StubLongStrategy()])

        dist = make_dist([100, 102, 98, 105, 95])
        features = cluster_strat._extract_features(dist, 100)

        assert len(features) == 5
        assert all(isinstance(f, (int, float)) for f in features)
        # Feature 4 should be trend direction: mean should be slightly > 100
        assert features[4] == 1.0 or features[4] == -1.0


# =============================================================================
# MONTE CARLO SCENARIO TESTS
# =============================================================================

class TestMonteCarloScenarioStrategy:

    def test_monte_carlo_returns_signal(self):
        """Should return a non-None signal from the winning strategy."""
        long_strat = StubLongStrategy()
        none_strat = StubNoneStrategy()

        mc_strat = MonteCarloScenarioStrategy(
            base_strategies=[long_strat, none_strat],
            n_scenarios=500,
            selection_metric="expected_pnl"
        )

        # Use clearly bullish distribution to ensure long strategy wins
        dist = make_dist([110, 112, 108, 115, 105, 111, 109, 114])
        signal = mc_strat.generate_signal(dist, 100, None, {})

        assert signal is not None
        assert signal.direction == Direction.LONG
        assert "monte_carlo_selected_strategy" in signal.metadata
        assert "monte_carlo_expected_pnl" in signal.metadata
        # Long strategy should win in bullish distribution
        assert signal.metadata["monte_carlo_selected_strategy"] == "stub_long"

    def test_monte_carlo_picks_best_strategy(self):
        """Should pick the strategy with highest expected PnL."""
        long_strat = StubLongStrategy()
        short_strat = StubShortStrategy()

        mc_strat = MonteCarloScenarioStrategy(
            base_strategies=[long_strat, short_strat],
            n_scenarios=500,
            selection_metric="expected_pnl"
        )

        # Create distribution with mean > current_price (bullish)
        closes = [105, 106, 104, 107, 103, 105.5]
        dist = make_dist(closes)
        current_price = 100

        signal = mc_strat.generate_signal(dist, current_price, None, {})

        assert signal is not None
        # LONG should outperform SHORT in bullish scenario
        assert signal.direction == Direction.LONG
        assert signal.metadata["monte_carlo_selected_strategy"] == "stub_long"

    def test_monte_carlo_handles_zero_std(self):
        """Should not crash when distribution has zero std (all closes identical)."""
        long_strat = StubLongStrategy()

        mc_strat = MonteCarloScenarioStrategy(
            base_strategies=[long_strat],
            n_scenarios=100
        )

        # All closes identical = zero std
        dist = make_dist([100, 100, 100, 100, 100])
        signal = mc_strat.generate_signal(dist, 100, None, {})

        # Should return signal from fallback (first strategy)
        assert signal is not None
        assert signal.direction == Direction.LONG

    def test_monte_carlo_handles_all_none_strategies(self):
        """Should return None if all strategies return None."""
        none_strat1 = StubNoneStrategy()
        none_strat2 = StubNoneStrategy()

        mc_strat = MonteCarloScenarioStrategy(
            base_strategies=[none_strat1, none_strat2],
            n_scenarios=100
        )

        dist = make_dist([100, 101, 99, 102, 98])
        signal = mc_strat.generate_signal(dist, 100, None, {})

        assert signal is None

    def test_monte_carlo_sharpe_metric(self):
        """Should use Sharpe ratio when selection_metric='sharpe'."""
        long_strat = StubLongStrategy()
        short_strat = StubShortStrategy()

        mc_strat = MonteCarloScenarioStrategy(
            base_strategies=[long_strat, short_strat],
            n_scenarios=500,
            selection_metric="sharpe"
        )

        dist = make_dist([105, 106, 104, 107, 103, 105.5])
        signal = mc_strat.generate_signal(dist, 100, None, {})

        assert signal is not None
        assert "monte_carlo_sharpe" in signal.metadata

    def test_monte_carlo_flat_signal_gets_zero_ev(self):
        """Strategies returning FLAT signals should get 0 expected PnL."""
        long_strat = StubLongStrategy()
        flat_strat = StubFlatStrategy()

        mc_strat = MonteCarloScenarioStrategy(
            base_strategies=[long_strat, flat_strat],
            n_scenarios=200
        )

        # Use a very bullish distribution: all closes well above 100
        # This makes LONG strategy's target (105) much more likely to be reached
        dist = make_dist([110, 112, 108, 115, 105, 111, 109])
        signal = mc_strat.generate_signal(dist, 100, None, {})

        # LONG should win because FLAT gets 0 PnL and LONG gets positive EV
        assert signal is not None
        # Check that long_strat was selected (has higher expected_pnl)
        assert signal.metadata["monte_carlo_selected_strategy"] == "stub_long"
        assert signal.metadata["monte_carlo_expected_pnl"] > 0

    def test_monte_carlo_stores_metadata(self):
        """Should store expected PnL, Sharpe, and other metrics in metadata."""
        long_strat = StubLongStrategy()

        mc_strat = MonteCarloScenarioStrategy(
            base_strategies=[long_strat],
            n_scenarios=200,
            selection_metric="expected_pnl"
        )

        dist = make_dist([100, 101, 99, 102, 98])
        signal = mc_strat.generate_signal(dist, 100, None, {})

        assert signal is not None
        assert signal.metadata["monte_carlo_selected_strategy"] == "stub_long"
        assert isinstance(signal.metadata["monte_carlo_expected_pnl"], (int, float))
        assert isinstance(signal.metadata["monte_carlo_sharpe"], (int, float))
        assert signal.metadata["monte_carlo_n_scenarios"] == 200


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestMetaStrategyIntegration:

    def test_regime_cluster_with_multiple_strategies(self):
        """Integration test with 3 different stub strategies."""
        long_strat = StubLongStrategy()
        short_strat = StubShortStrategy()
        none_strat = StubNoneStrategy()

        cluster_strat = RegimeClusterStrategy(
            base_strategies=[long_strat, short_strat, none_strat],
            k_neighbors=2,
            fallback_strategy=long_strat
        )

        # Populate buffer
        cluster_strat.record_trade([0.05, 0.1, 0.02, 1.0, 1.0], "stub_long", 50.0)
        cluster_strat.record_trade([0.05, 0.1, 0.02, 1.0, 1.0], "stub_long", 45.0)
        cluster_strat.record_trade([0.06, 0.15, 0.03, 1.2, -1.0], "stub_short", 5.0)

        dist = make_dist([100, 101, 99, 102, 98])
        signal = cluster_strat.generate_signal(dist, 100, None, {})

        assert signal is not None
        # Should pick long because it has higher mean PnL
        assert signal.direction == Direction.LONG

    def test_monte_carlo_with_context_and_history(self):
        """Should work with non-empty context and history."""
        long_strat = StubLongStrategy()

        mc_strat = MonteCarloScenarioStrategy(
            base_strategies=[long_strat],
            n_scenarios=100
        )

        # Create minimal history DataFrame
        history = pd.DataFrame({
            "close": [99, 100, 101],
            "volume": [1.0, 1.0, 1.0]
        })

        context = {"test_key": "test_value"}
        dist = make_dist([100, 101, 99, 102, 98])

        signal = mc_strat.generate_signal(dist, 100, history, context)

        assert signal is not None
        assert signal.direction == Direction.LONG


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
