"""
Unit tests for ConditionalPathProbabilityStrategy

Tests the strategy's ability to:
1. Detect range days (high p_range) and emit FLAT sell_straddle signals
2. Detect trend days (low p_range) and emit directional signals
3. Return None for ambiguous days (p_range between thresholds)
4. Use raw sample iteration (not aggregated stats)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../strategy"))

import pandas as pd
import numpy as np
from kairos_backtest import KairosDistribution, Direction, Signal
from kairos_path import ConditionalPathProbabilityStrategy


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def make_dist(closes, highs=None, lows=None):
    """
    Create a KairosDistribution with controllable high/low samples.

    Args:
        closes: List of close prices (one per sample)
        highs: List of high prices. If None, defaults to close * 1.01
        lows: List of low prices. If None, defaults to close * 0.99

    Returns:
        KairosDistribution instance
    """
    rows = []
    n = len(closes)

    if highs is None:
        highs = [c * 1.01 for c in closes]
    if lows is None:
        lows = [c * 0.99 for c in closes]

    for i in range(n):
        c = closes[i]
        h = highs[i]
        l = lows[i]
        rows.append(pd.DataFrame({
            "open": [c],
            "high": [h],
            "low": [l],
            "close": [c],
            "volume": [1.0]
        }))

    return KairosDistribution(rows)


# =============================================================================
# TEST 1: Range Day (p_range = 1.0)
# =============================================================================

def test_range_day_returns_flat_sell_straddle():
    """
    Range day: every sample's high >= pred_high AND low <= pred_low.
    Expected: Signal with direction=FLAT, metadata["action"]="sell_straddle"
    """
    # Create 60 samples where:
    # - All highs are 102.0 (= pred_high)
    # - All lows are 98.0 (= pred_low)
    # - Closes vary slightly around 100.0
    closes = [100.0 + np.random.normal(0, 0.1) for _ in range(60)]
    highs = [102.0 for _ in range(60)]
    lows = [98.0 for _ in range(60)]

    dist = make_dist(closes, highs, lows)

    # With these samples, pred_high should be ~102.0 and pred_low should be ~98.0
    # Every sample hits both, so p_range should be 1.0
    strategy = ConditionalPathProbabilityStrategy(range_threshold=0.70, trend_threshold=0.30)
    current_price = 100.0
    signal = strategy.generate_signal(dist, current_price, None, {})

    # Should return a signal (not None)
    assert signal is not None, "Range day should return a signal"

    # Should be FLAT
    assert signal.direction == Direction.FLAT, f"Expected FLAT, got {signal.direction}"

    # Should have sell_straddle metadata
    assert "action" in signal.metadata, "Signal should have 'action' in metadata"
    assert signal.metadata["action"] == "sell_straddle", \
        f"Expected 'sell_straddle', got {signal.metadata['action']}"

    # Confidence should be close to 1.0
    assert signal.confidence >= 0.95, f"Expected high confidence, got {signal.confidence}"

    print("✓ test_range_day_returns_flat_sell_straddle PASSED")


# =============================================================================
# TEST 2: Trend Day (p_range = 0.0)
# =============================================================================

def test_trend_day_returns_directional():
    """
    Trend day: very few samples hit both high and low (p_range << 0.30).
    Expected: Directional signal (LONG or SHORT based on close mean vs current price)

    Strategy: First 30 samples are "high" cluster (high=105, low=104).
              Last 30 samples are "low" cluster (high=104, low=103).
              pred_high = (105*30 + 104*30)/60 = 104.5
              pred_low = (104*30 + 103*30)/60 = 103.5
              High cluster: high >= 104.5 (YES) but low <= 103.5 (NO) -> miss
              Low cluster: low <= 103.5 (YES) but high >= 104.5 (NO) -> miss
              p_range = 0.0 (no sample hits both)
    """
    closes = [103.5 for _ in range(60)]

    # High cluster: 30 samples staying near top
    highs = [105.0] * 30 + [104.0] * 30
    lows = [104.0] * 30 + [103.0] * 30

    dist = make_dist(closes, highs, lows)

    strategy = ConditionalPathProbabilityStrategy(range_threshold=0.70, trend_threshold=0.30)
    current_price = 100.0
    signal = strategy.generate_signal(dist, current_price, None, {})

    # Should return a signal (not None)
    assert signal is not None, "Trend day should return a signal"

    # Should NOT be FLAT
    assert signal.direction != Direction.FLAT, "Trend day should not be FLAT"

    # close_mean (103.5) > current_price (100.0), so should be LONG
    assert signal.direction == Direction.LONG, \
        f"Expected LONG (close_mean > current_price), got {signal.direction}"

    # Confidence should be 1 - p_range (high, p_range should be 0.0)
    assert signal.confidence >= 0.95, f"Expected high confidence, got {signal.confidence}"

    # p_range should be very close to 0.0
    assert signal.metadata["p_range"] < 0.05, \
        f"Expected p_range < 0.05, got {signal.metadata['p_range']}"

    print("✓ test_trend_day_returns_directional PASSED")


# =============================================================================
# TEST 3: Ambiguous (p_range between thresholds)
# =============================================================================

def test_ambiguous_returns_none():
    """
    Ambiguous day: p_range between trend_threshold and range_threshold.
    Expected: return None
    """
    # Create 60 samples where exactly 30 out of 60 hit both extremes
    # p_range = 0.5, which is between 0.30 and 0.70
    closes = [100.0 for _ in range(60)]

    # First 30 samples: hit both extremes (102.0 and 98.0)
    highs_hit = [102.0 for _ in range(30)]
    lows_hit = [98.0 for _ in range(30)]

    # Last 30 samples: only hit high (101.0), not low (101.0)
    highs_miss = [101.0 for _ in range(30)]
    lows_miss = [101.0 for _ in range(30)]

    highs = highs_hit + highs_miss
    lows = lows_hit + lows_miss

    dist = make_dist(closes, highs, lows)

    strategy = ConditionalPathProbabilityStrategy(range_threshold=0.70, trend_threshold=0.30)
    current_price = 100.0
    signal = strategy.generate_signal(dist, current_price, None, {})

    # Should return None for ambiguous
    assert signal is None, f"Expected None for ambiguous day, got {signal}"

    print("✓ test_ambiguous_returns_none PASSED")


# =============================================================================
# TEST 4: Uses raw samples, not aggregated stats
# =============================================================================

def test_uses_raw_samples_not_aggregated():
    """
    Verify that the count is computed by iterating dist.predictions,
    not by using aggregated dist.stats.

    This is critical because dist.stats might be aggregated differently
    (e.g., mean vs percentile) and could give different results.
    """
    # Create 60 samples with a specific pattern:
    # - All have high=102.0 and low=98.0
    # - Closes vary
    # This ensures p_range = 1.0 from raw sample iteration

    closes = [100.0 + i * 0.01 for i in range(60)]
    highs = [102.0 for _ in range(60)]
    lows = [98.0 for _ in range(60)]

    dist = make_dist(closes, highs, lows)

    # Now manually compute what aggregated stats would be
    # Stats should have:
    # - high.mean ≈ 102.0
    # - low.mean ≈ 98.0
    # - high.pct_90 and low.pct_10 based on aggregated close values

    assert dist.stats["high"]["mean"] == 102.0, "High mean should be exactly 102.0"
    assert dist.stats["low"]["mean"] == 98.0, "Low mean should be exactly 98.0"

    strategy = ConditionalPathProbabilityStrategy(range_threshold=0.70, trend_threshold=0.30)
    current_price = 100.0
    signal = strategy.generate_signal(dist, current_price, None, {})

    # Signal should be FLAT (range day, p_range = 1.0)
    assert signal is not None, "Should return a signal"
    assert signal.direction == Direction.FLAT, "Should be FLAT for p_range=1.0"
    assert signal.confidence == 1.0, f"Confidence should be 1.0, got {signal.confidence}"

    print("✓ test_uses_raw_samples_not_aggregated PASSED")


# =============================================================================
# ADDITIONAL EDGE CASE TESTS
# =============================================================================

def test_threshold_boundaries():
    """Test behavior exactly at threshold boundaries."""
    # Create a distribution with p_range exactly at range_threshold (0.70)
    # 42 out of 60 samples hit both (42/60 = 0.7)
    closes = [100.0 for _ in range(60)]
    highs_hit = [102.0 for _ in range(42)]
    highs_miss = [100.0 for _ in range(18)]
    lows_hit = [98.0 for _ in range(42)]
    lows_miss = [100.0 for _ in range(18)]

    highs = highs_hit + highs_miss
    lows = lows_hit + lows_miss

    dist = make_dist(closes, highs, lows)

    strategy = ConditionalPathProbabilityStrategy(range_threshold=0.70, trend_threshold=0.30)
    current_price = 100.0
    signal = strategy.generate_signal(dist, current_price, None, {})

    # At exactly 0.70, p_range > 0.70 is False, so should not be FLAT
    # And p_range < 0.30 is False, so should not be trend
    # Should return None
    assert signal is None, "At threshold boundary, should return None"

    print("✓ test_threshold_boundaries PASSED")


def test_short_trend_day():
    """Test trend day with SHORT signal when close_mean < current_price."""
    rng = np.random.default_rng(42)
    # Closes well below pred_high (105) and pred_low (92) so p_range stays < 0.30
    closes = [97.0 + rng.normal(0, 0.5) for _ in range(60)]
    highs = [98.0 + rng.normal(0, 0.5) for _ in range(60)]   # mean ~98, pred_high ~98
    lows = [95.0 + rng.normal(0, 0.5) for _ in range(60)]    # mean ~95, pred_low ~95

    dist = make_dist(closes, highs, lows)

    strategy = ConditionalPathProbabilityStrategy(range_threshold=0.70, trend_threshold=0.30)
    current_price = 100.0  # Current is above the trend
    signal = strategy.generate_signal(dist, current_price, None, {})

    # close_mean (~97.0) < current_price (100.0), so should be SHORT
    assert signal is not None, "Should return a signal for trend day"
    assert signal.direction == Direction.SHORT, \
        f"Expected SHORT (close_mean < current_price), got {signal.direction}"

    print("✓ test_short_trend_day PASSED")


if __name__ == "__main__":
    test_range_day_returns_flat_sell_straddle()
    test_trend_day_returns_directional()
    test_ambiguous_returns_none()
    test_uses_raw_samples_not_aggregated()
    test_threshold_boundaries()
    test_short_trend_day()
    print("\n✓ All tests PASSED!")
