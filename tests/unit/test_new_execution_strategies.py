"""
test_new_execution_strategies.py
=================================
Unit tests for PyramidingStrategy and TimeBasedStopStrategy.
"""

import sys
import os

# Add strategy/ to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../strategy"))

import pandas as pd
import numpy as np
from kairos_backtest import KairosDistribution, Direction, Signal, Strategy


# =============================================================================
# HELPERS
# =============================================================================

def make_dist(closes, highs=None, lows=None):
    """Create a KairosDistribution from close prices."""
    if highs is None:
        highs = [c * 1.01 for c in closes]
    if lows is None:
        lows = [c * 0.99 for c in closes]

    rows = []
    for i, c in enumerate(closes):
        row = pd.DataFrame({
            "open": [c],
            "high": [highs[i]],
            "low": [lows[i]],
            "close": [c],
            "volume": [1.0]
        })
        rows.append(row)

    return KairosDistribution(rows)


class StubStrategy(Strategy):
    """A test strategy that returns a pre-configured signal."""
    name = "stub"

    def __init__(self, sig):
        self._sig = sig

    def generate_signal(self, dist, current_price, history, context, **kwargs):
        return self._sig


# =============================================================================
# TESTS FOR PyramidingStrategy
# =============================================================================

def test_pyramiding_adds_plan_to_metadata():
    """Test that PyramidingStrategy adds pyramid_plan to metadata."""
    from kairos_execution import PyramidingStrategy

    # Create a LONG signal
    signal = Signal(
        direction=Direction.LONG,
        size=0.5,
        entry=100.0,
        stop=95.0,
        target=110.0,
        strategy_name="stub",
        confidence=0.8,
        expected_value=2.5,
        metadata={}
    )

    stub = StubStrategy(signal)
    strat = PyramidingStrategy(
        base_strategy=stub,
        pyramid_threshold_pct=0.01,
        pyramid_add_pct=0.25,
        max_pyramid_levels=3
    )

    dist = make_dist([100, 101, 102])
    result = strat.generate_signal(dist, 100.0, pd.DataFrame(), {})

    # Check that pyramid_plan was added
    assert "pyramid_plan" in result.metadata
    plan = result.metadata["pyramid_plan"]
    assert len(plan) == 3

    # Check structure of first level
    assert plan[0]["level"] == 1
    assert plan[0]["add_size"] == 0.5 * 0.25
    assert plan[0]["stop"] == 95.0
    assert plan[0]["target"] == 110.0

    # For LONG, each level's add_at_price should be > entry
    for level_plan in plan:
        assert level_plan["add_at_price"] > 100.0


def test_pyramiding_levels_increase_for_long():
    """Test that pyramid levels increase correctly for LONG."""
    from kairos_execution import PyramidingStrategy

    signal = Signal(
        direction=Direction.LONG,
        size=0.5,
        entry=100.0,
        stop=95.0,
        target=110.0,
        strategy_name="stub",
        confidence=0.8,
        expected_value=2.5,
        metadata={}
    )

    stub = StubStrategy(signal)
    strat = PyramidingStrategy(
        base_strategy=stub,
        pyramid_threshold_pct=0.01,
        pyramid_add_pct=0.25,
        max_pyramid_levels=3
    )

    dist = make_dist([100, 101, 102])
    result = strat.generate_signal(dist, 100.0, pd.DataFrame(), {})

    plan = result.metadata["pyramid_plan"]
    # Each level should be at a higher price for LONG
    for i in range(len(plan) - 1):
        assert plan[i]["add_at_price"] < plan[i + 1]["add_at_price"]


def test_pyramiding_levels_decrease_for_short():
    """Test that pyramid levels decrease correctly for SHORT."""
    from kairos_execution import PyramidingStrategy

    signal = Signal(
        direction=Direction.SHORT,
        size=0.5,
        entry=100.0,
        stop=105.0,
        target=90.0,
        strategy_name="stub",
        confidence=0.8,
        expected_value=2.5,
        metadata={}
    )

    stub = StubStrategy(signal)
    strat = PyramidingStrategy(
        base_strategy=stub,
        pyramid_threshold_pct=0.01,
        pyramid_add_pct=0.25,
        max_pyramid_levels=3
    )

    dist = make_dist([100, 99, 98])
    result = strat.generate_signal(dist, 100.0, pd.DataFrame(), {})

    plan = result.metadata["pyramid_plan"]
    # Each level should be at a lower price for SHORT
    for i in range(len(plan) - 1):
        assert plan[i]["add_at_price"] > plan[i + 1]["add_at_price"]


def test_pyramiding_returns_none_if_base_none():
    """Test that PyramidingStrategy returns None if base strategy returns None."""
    from kairos_execution import PyramidingStrategy

    stub = StubStrategy(None)
    strat = PyramidingStrategy(base_strategy=stub)

    dist = make_dist([100, 101, 102])
    result = strat.generate_signal(dist, 100.0, pd.DataFrame(), {})

    assert result is None


def test_pyramiding_passes_flat_through():
    """Test that PyramidingStrategy passes FLAT signals through without pyramid plan."""
    from kairos_execution import PyramidingStrategy

    signal = Signal(
        direction=Direction.FLAT,
        size=0.0,
        entry=100.0,
        stop=95.0,
        target=110.0,
        strategy_name="stub",
        confidence=0.0,
        expected_value=0.0,
        metadata={}
    )

    stub = StubStrategy(signal)
    strat = PyramidingStrategy(base_strategy=stub)

    dist = make_dist([100, 101, 102])
    result = strat.generate_signal(dist, 100.0, pd.DataFrame(), {})

    assert result is not None
    assert result.direction == Direction.FLAT
    # FLAT signals should not have pyramid_plan added
    assert "pyramid_plan" not in result.metadata or len(result.metadata.get("pyramid_plan", [])) == 0


# =============================================================================
# TESTS FOR TimeBasedStopStrategy
# =============================================================================

def test_time_stop_adds_metadata():
    """Test that TimeBasedStopStrategy adds time-exit metadata."""
    from kairos_execution import TimeBasedStopStrategy

    signal = Signal(
        direction=Direction.LONG,
        size=0.5,
        entry=100.0,
        stop=95.0,
        target=110.0,
        strategy_name="stub",
        confidence=0.8,
        expected_value=2.5,
        metadata={}
    )

    stub = StubStrategy(signal)
    strat = TimeBasedStopStrategy(base_strategy=stub, time_bars=2, exit_at="close")

    dist = make_dist([100, 101, 102])
    context = {"bar_index": 5}
    result = strat.generate_signal(dist, 100.0, pd.DataFrame(), context)

    assert result is not None
    assert result.metadata["time_exit_enabled"] is True
    assert result.metadata["time_exit_bar"] == 7  # 5 + 2
    assert result.metadata["time_exit_price"] == 100.0  # current_price


def test_time_stop_default_bar_index():
    """Test that TimeBasedStopStrategy defaults bar_index to 0."""
    from kairos_execution import TimeBasedStopStrategy

    signal = Signal(
        direction=Direction.LONG,
        size=0.5,
        entry=100.0,
        stop=95.0,
        target=110.0,
        strategy_name="stub",
        confidence=0.8,
        expected_value=2.5,
        metadata={}
    )

    stub = StubStrategy(signal)
    strat = TimeBasedStopStrategy(base_strategy=stub, time_bars=1)

    dist = make_dist([100, 101, 102])
    context = {}  # No bar_index
    result = strat.generate_signal(dist, 100.0, pd.DataFrame(), context)

    assert result is not None
    assert result.metadata["time_exit_bar"] == 1  # 0 + 1


def test_time_stop_predicted_median():
    """Test that TimeBasedStopStrategy uses predicted median when exit_at='predicted_median'."""
    from kairos_execution import TimeBasedStopStrategy

    signal = Signal(
        direction=Direction.LONG,
        size=0.5,
        entry=100.0,
        stop=95.0,
        target=110.0,
        strategy_name="stub",
        confidence=0.8,
        expected_value=2.5,
        metadata={}
    )

    stub = StubStrategy(signal)
    strat = TimeBasedStopStrategy(base_strategy=stub, time_bars=1, exit_at="predicted_median")

    closes = [100, 101, 102, 103, 104]
    dist = make_dist(closes)
    context = {"bar_index": 0}
    result = strat.generate_signal(dist, 100.0, pd.DataFrame(), context)

    assert result is not None
    # predicted_median should be the 50th percentile
    assert result.metadata["time_exit_price"] == dist.stats["close"]["pct_50"]


def test_time_stop_none_if_base_none():
    """Test that TimeBasedStopStrategy returns None if base strategy returns None."""
    from kairos_execution import TimeBasedStopStrategy

    stub = StubStrategy(None)
    strat = TimeBasedStopStrategy(base_strategy=stub, time_bars=1)

    dist = make_dist([100, 101, 102])
    context = {"bar_index": 5}
    result = strat.generate_signal(dist, 100.0, pd.DataFrame(), context)

    assert result is None


def test_time_stop_preserves_base_signal():
    """Test that TimeBasedStopStrategy preserves base signal properties."""
    from kairos_execution import TimeBasedStopStrategy

    signal = Signal(
        direction=Direction.SHORT,
        size=0.3,
        entry=100.0,
        stop=105.0,
        target=90.0,
        strategy_name="stub",
        confidence=0.75,
        expected_value=1.5,
        metadata={"custom": "data"}
    )

    stub = StubStrategy(signal)
    strat = TimeBasedStopStrategy(base_strategy=stub, time_bars=3)

    dist = make_dist([100, 99, 98])
    context = {"bar_index": 10}
    result = strat.generate_signal(dist, 100.0, pd.DataFrame(), context)

    # Base signal properties should be preserved
    assert result.direction == Direction.SHORT
    assert result.size == 0.3
    assert result.entry == 100.0
    assert result.stop == 105.0
    assert result.target == 90.0
    assert result.strategy_name == "stub"
    assert result.confidence == 0.75
    assert result.expected_value == 1.5
    # Custom metadata should be preserved
    assert result.metadata["custom"] == "data"
    # Time-exit metadata should be added
    assert result.metadata["time_exit_enabled"] is True
    assert result.metadata["time_exit_bar"] == 13  # 10 + 3


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
