"""
test_path_integration.py
=======================

Unit tests for PathIntegrationStrategy.
Tests the multi-horizon path quality analysis and signal generation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../strategy"))

import pandas as pd
import numpy as np
import pytest
from kairos_backtest import KairosDistribution, Direction, Signal
from kairos_horizon import HorizonStack, PathIntegrationStrategy


def make_dist(closes, std_multiplier=0.01):
    """
    Helper to create a KairosDistribution from close prices.
    std_multiplier controls the spread around each close price.
    """
    rows = []
    for c in closes:
        df = pd.DataFrame({
            "open": [c],
            "high": [c * (1 + std_multiplier)],
            "low": [c * (1 - std_multiplier)],
            "close": [c],
            "volume": [1.0],
        })
        rows.append(df)
    return KairosDistribution(rows)


class TestPathIntegrationStrategy:
    """Test suite for PathIntegrationStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create a PathIntegrationStrategy instance."""
        return PathIntegrationStrategy(
            max_horizon=3,
            variance_tightening_threshold=0.9,
            entropy_spike_threshold=1.5
        )

    def test_no_horizon_stack_returns_none(self, strategy):
        """
        Test: When context has no horizon_stack, signal is None.
        """
        current_price = 100.0
        closes = [100.0 + i * 0.5 for i in range(60)]
        dist = make_dist(closes)
        history = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000.0],
        })
        context = {}  # No horizon_stack

        signal = strategy.generate_signal(dist, current_price, history, context)
        assert signal is None

    def test_consistent_tightening_returns_3_days(self, strategy):
        """
        Test: Consistent direction + variance tightening across horizons -> 3 days.

        Setup:
        - H1: mean=101 (LONG), std=2.0
        - H2: mean=102 (LONG), std=1.5 (tightens from H1)
        - H3: mean=103 (LONG), std=1.0 (tightens from H2)
        - All means > base_price=100 (consistent LONG)
        - std tightening: 1.5 < 2.0*0.9=1.8, 1.0 < 1.5*0.9=1.35
        """
        current_price = 100.0
        base_price = 100.0

        # Build distributions with exact means and exact stds using two-point
        # distributions: [v_lo]*30 + [v_hi]*30 gives mean=(v_lo+v_hi)/2,
        # std=(v_hi-v_lo)/2 deterministically — no random sampling, no flakiness.
        # H1: mean=101, std=2  (v_lo=99, v_hi=103)
        h1_closes = np.concatenate([np.full(30, 99.0), np.full(30, 103.0)])
        h1_dist = make_dist(h1_closes, std_multiplier=0.02)

        # H2: mean=102, std=1.5  (v_lo=100.5, v_hi=103.5)
        h2_closes = np.concatenate([np.full(30, 100.5), np.full(30, 103.5)])
        h2_dist = make_dist(h2_closes, std_multiplier=0.015)

        # H3: mean=103, std=1.0  (v_lo=102, v_hi=104)
        h3_closes = np.concatenate([np.full(30, 102.0), np.full(30, 104.0)])
        h3_dist = make_dist(h3_closes, std_multiplier=0.01)

        # Create HorizonStack
        stack = HorizonStack(
            horizons={1: h1_dist, 2: h2_dist, 3: h3_dist},
            base_price=base_price
        )

        # Verify the tightening
        s1 = h1_dist.stats["close"]
        s2 = h2_dist.stats["close"]
        s3 = h3_dist.stats["close"]
        assert s2["std"] < s1["std"] * 0.9
        assert s3["std"] < s2["std"] * 0.9

        # Build context
        history = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000.0],
        })
        context = {"horizon_stack": stack}

        # Generate signal
        signal = strategy.generate_signal(h1_dist, current_price, history, context)

        # Assertions
        assert signal is not None, "Should generate a signal"
        assert signal.direction == Direction.LONG, "Should be LONG (mean > current)"
        assert signal.metadata["hold_days"] == 3, "Should hold 3 days (consistent + tightening)"
        assert signal.confidence == 0.9, "Confidence should be 0.9"

    def test_consistent_no_tightening_returns_2_days(self, strategy):
        """
        Test: Consistent direction, no tightening -> 2 days.

        Setup:
        - H1: mean=101, std=2.0
        - H2: mean=102, std=2.0 (NOT tightening)
        - H3: mean=103, std=2.0 (NOT tightening)
        - All means > base_price=100 (consistent)
        """
        current_price = 100.0
        base_price = 100.0

        # All with same std (no tightening)
        h1_closes = np.random.normal(101.0, 2.0, 60)
        h1_dist = make_dist(h1_closes, std_multiplier=0.02)

        h2_closes = np.random.normal(102.0, 2.0, 60)
        h2_dist = make_dist(h2_closes, std_multiplier=0.02)

        h3_closes = np.random.normal(103.0, 2.0, 60)
        h3_dist = make_dist(h3_closes, std_multiplier=0.02)

        stack = HorizonStack(
            horizons={1: h1_dist, 2: h2_dist, 3: h3_dist},
            base_price=base_price
        )

        history = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000.0],
        })
        context = {"horizon_stack": stack}

        signal = strategy.generate_signal(h1_dist, current_price, history, context)

        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.metadata["hold_days"] == 2, "Consistent but no tightening -> 2 days"
        assert signal.confidence == 0.7, "Confidence should be 0.7"

    def test_inconsistent_returns_1_day(self, strategy):
        """
        Test: Inconsistent direction across horizons -> 1 day.

        Setup:
        - H1: mean=101 (LONG), base=100
        - H2: mean=99 (SHORT) -> direction breaks early
        - H3: mean=98 (SHORT)
        """
        current_price = 100.0
        base_price = 100.0

        # H1 is LONG, but H2 flips to SHORT immediately
        h1_closes = np.random.normal(101.0, 1.5, 60)
        h1_dist = make_dist(h1_closes, std_multiplier=0.015)

        h2_closes = np.random.normal(99.0, 1.5, 60)  # Direction flips
        h2_dist = make_dist(h2_closes, std_multiplier=0.015)

        h3_closes = np.random.normal(98.0, 1.5, 60)  # Also SHORT
        h3_dist = make_dist(h3_closes, std_multiplier=0.015)

        stack = HorizonStack(
            horizons={1: h1_dist, 2: h2_dist, 3: h3_dist},
            base_price=base_price
        )

        history = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000.0],
        })
        context = {"horizon_stack": stack}

        signal = strategy.generate_signal(h1_dist, current_price, history, context)

        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.metadata["hold_days"] == 1, "Inconsistent path (d1!=d2) -> 1 day"
        assert signal.confidence == 0.3, "Confidence should be 0.3"

    def test_missing_horizon_1_returns_none(self, strategy):
        """
        Test: If horizon_stack.horizons[1] is missing, return None.

        Setup:
        - horizons only contains {2, 3}, missing key 1
        """
        current_price = 100.0
        base_price = 100.0

        # Only create H2 and H3, skip H1
        h2_closes = np.random.normal(102.0, 1.5, 60)
        h2_dist = make_dist(h2_closes)

        h3_closes = np.random.normal(103.0, 1.0, 60)
        h3_dist = make_dist(h3_closes)

        stack = HorizonStack(
            horizons={2: h2_dist, 3: h3_dist},  # Missing key 1
            base_price=base_price
        )

        history = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000.0],
        })
        context = {"horizon_stack": stack}

        # Need a dummy dist for the call
        dummy_closes = np.random.normal(100.0, 1.0, 60)
        dummy_dist = make_dist(dummy_closes)

        signal = strategy.generate_signal(dummy_dist, current_price, history, context)
        assert signal is None, "Should return None if H1 is missing"

    def test_partially_consistent_direction(self, strategy):
        """
        Test: H1==H2 but H2!=H3 -> 2 days, confidence 0.5.

        Setup:
        - H1: mean=101 (LONG)
        - H2: mean=102 (LONG)
        - H3: mean=99 (SHORT)
        """
        current_price = 100.0
        base_price = 100.0

        h1_closes = np.random.normal(101.0, 1.5, 60)
        h1_dist = make_dist(h1_closes)

        h2_closes = np.random.normal(102.0, 1.5, 60)
        h2_dist = make_dist(h2_closes)

        h3_closes = np.random.normal(99.0, 1.5, 60)
        h3_dist = make_dist(h3_closes)

        stack = HorizonStack(
            horizons={1: h1_dist, 2: h2_dist, 3: h3_dist},
            base_price=base_price
        )

        history = pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [1000.0],
        })
        context = {"horizon_stack": stack}

        signal = strategy.generate_signal(h1_dist, current_price, history, context)

        assert signal is not None
        assert signal.direction == Direction.LONG
        # d1 == d2 and d2 != d3 -> 2 days
        assert signal.metadata["hold_days"] == 2, "H1==H2 but H2!=H3 -> 2 days"
        assert signal.confidence == 0.5, "Confidence should be 0.5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
