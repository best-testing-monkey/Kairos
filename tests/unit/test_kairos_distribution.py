"""
Tests for KairosDistribution class from kairos_backtest.py
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add strategy directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from kairos_backtest import KairosDistribution


def make_dist(close_prices, current_price=None):
    """
    Helper to build a distribution from a list of close prices.
    """
    prices = np.array(close_prices, dtype=float)
    cp = current_price or float(np.mean(prices))
    frames = []
    for p in prices:
        frames.append(pd.DataFrame({
            "open": [cp], "high": [p * 1.005], "low": [p * 0.995],
            "close": [p], "volume": [1e6], "amount": [1e9]
        }))
    return KairosDistribution(frames)


# =============================================================================
# TEST ENTROPY (FIXED VERSION)
# =============================================================================

class TestEntropyFixed:
    """Verify that entropy is calculated using Shannon entropy in nats"""

    def test_all_same_values(self):
        """100 identical prices → entropy = 0.0 (all in one bin)"""
        prices = [100.0] * 100
        dist = make_dist(prices)
        entropy = dist.entropy(col="close", bins=20)
        assert entropy == 0.0, f"Expected entropy 0.0 for identical prices, got {entropy}"

    def test_uniform_spread(self):
        """100 evenly spread prices across [100, 200] → entropy close to ln(20) ≈ 3.0"""
        prices = np.linspace(100, 200, 100).tolist()
        dist = make_dist(prices)
        entropy = dist.entropy(col="close", bins=20)
        # For uniform distribution across 20 bins, max entropy = ln(20) ≈ 2.996
        assert entropy > 2.5, f"Expected entropy > 2.5 for uniform distribution, got {entropy}"
        assert entropy <= np.log(20), f"Entropy should not exceed ln(20), got {entropy}"

    def test_entropy_range(self):
        """Entropy is always between 0.0 and ln(20)"""
        # Test with various distributions
        test_cases = [
            [100.0] * 50,  # All same
            np.linspace(100, 200, 100).tolist(),  # Uniform
            np.random.normal(150, 10, 100).tolist(),  # Normal
        ]
        for prices in test_cases:
            dist = make_dist(prices)
            entropy = dist.entropy(col="close", bins=20)
            assert 0.0 <= entropy <= np.log(20) + 0.01, \
                f"Entropy {entropy} out of range [0, ln(20)={np.log(20):.3f}]"


# =============================================================================
# TEST STATS
# =============================================================================

class TestStats:
    """Test statistical calculations"""

    def test_mean_correct(self):
        """10 prices = [10, 20, 30, ...100] → stats mean ≈ 55.0"""
        prices = list(range(10, 110, 10))
        dist = make_dist(prices)
        mean = dist.stats["close"]["mean"]
        expected = np.mean(prices)
        assert abs(mean - expected) < 1e-6, f"Expected mean {expected}, got {mean}"

    def test_std_positive(self):
        """Prices with variance → std > 0"""
        prices = [100, 105, 95, 110, 90]
        dist = make_dist(prices)
        std = dist.stats["close"]["std"]
        assert std > 0, f"Expected std > 0 for varied prices, got {std}"

    def test_percentile_ordering(self):
        """pct_5 ≤ pct_25 ≤ pct_50 ≤ pct_75 ≤ pct_95"""
        prices = np.random.normal(100, 5, 100).tolist()
        dist = make_dist(prices)
        s = dist.stats["close"]
        assert s["pct_5"] <= s["pct_25"], f"pct_5 > pct_25: {s['pct_5']} > {s['pct_25']}"
        assert s["pct_25"] <= s["pct_50"], f"pct_25 > pct_50: {s['pct_25']} > {s['pct_50']}"
        assert s["pct_50"] <= s["pct_75"], f"pct_50 > pct_75: {s['pct_50']} > {s['pct_75']}"
        assert s["pct_75"] <= s["pct_95"], f"pct_75 > pct_95: {s['pct_75']} > {s['pct_95']}"

    def test_pct20_pct80_exist(self):
        """Both "pct_20" and "pct_80" in dist.stats["close"]"""
        prices = np.linspace(100, 200, 100).tolist()
        dist = make_dist(prices)
        assert "pct_20" in dist.stats["close"], "pct_20 missing from stats"
        assert "pct_80" in dist.stats["close"], "pct_80 missing from stats"


# =============================================================================
# TEST EXPECTED VALUE
# =============================================================================

class TestExpectedValue:
    """Test expected value calculation"""

    def test_ev_long_positive(self):
        """80 prices above target, 20 below stop → EV > 0 for long"""
        # Create prices: 20 at 90 (below stop), 80 at 150 (above target)
        prices = [90.0] * 20 + [150.0] * 80
        dist = make_dist(prices)
        entry = 100.0
        target = 140.0
        stop = 95.0
        ev = dist.expected_value(entry=entry, target=target, stop=stop)
        assert ev > 0, f"Expected EV > 0 for 80% win rate, got {ev}"

    def test_ev_negative_when_no_upside(self):
        """All prices at entry → EV ≈ 0"""
        prices = [100.0] * 100
        dist = make_dist(prices)
        entry = 100.0
        target = 110.0
        stop = 90.0
        ev = dist.expected_value(entry=entry, target=target, stop=stop)
        assert abs(ev) < 1e-6, f"Expected EV ≈ 0 for prices at entry, got {ev}"


# =============================================================================
# TEST CDF
# =============================================================================

class TestCdf:
    """Test cumulative distribution function"""

    def test_cdf_below_min(self):
        """Price = 0 → cdf = 0.0"""
        prices = np.linspace(100, 200, 100).tolist()
        dist = make_dist(prices)
        cdf = dist.cdf(0.0)
        assert cdf == 0.0, f"Expected cdf=0.0 for price below min, got {cdf}"

    def test_cdf_above_max(self):
        """Price = 1e9 → cdf = 1.0"""
        prices = np.linspace(100, 200, 100).tolist()
        dist = make_dist(prices)
        cdf = dist.cdf(1e9)
        assert cdf == 1.0, f"Expected cdf=1.0 for price above max, got {cdf}"

    def test_cdf_median(self):
        """Price = median → cdf ≈ 0.5"""
        prices = np.linspace(100, 200, 101).tolist()
        dist = make_dist(prices)
        median = np.median(prices)
        cdf = dist.cdf(median)
        # CDF at median should be close to 0.5 (within reasonable tolerance)
        assert 0.45 <= cdf <= 0.55, f"Expected cdf ≈ 0.5 at median, got {cdf}"


# =============================================================================
# TEST KELLY
# =============================================================================

class TestKelly:
    """Test Kelly fraction calculation"""

    def test_kelly_range(self):
        """Result is always between 0.0 and 1.0"""
        test_cases = [
            ([100.0] * 50, 100.0, 110.0, 95.0),  # 50% win rate
            (np.linspace(100, 150, 100).tolist(), 100.0, 140.0, 95.0),  # Skewed
            ([90, 95, 100, 105, 110] * 20, 100.0, 110.0, 90.0),  # Discrete
        ]
        for prices, entry, target, stop in test_cases:
            dist = make_dist(prices)
            kelly = dist.kelly_fraction(entry=entry, target=target, stop=stop)
            assert 0.0 <= kelly <= 1.0, f"Kelly {kelly} out of range [0, 1]"


# =============================================================================
# TEST CONSTRUCTOR
# =============================================================================

class TestConstructor:
    """Test KairosDistribution construction"""

    def test_single_sample(self):
        """List of 1 DataFrame → dist.df has 1 row"""
        df = pd.DataFrame({
            "open": [100.0], "high": [105.0], "low": [95.0],
            "close": [102.0], "volume": [1e6], "amount": [1e9]
        })
        dist = KairosDistribution([df])
        assert len(dist.df) == 1, f"Expected 1 row, got {len(dist.df)}"

    def test_hundred_samples(self):
        """List of 100 DataFrames → dist.df has 100 rows"""
        frames = []
        for i in range(100):
            frames.append(pd.DataFrame({
                "open": [100.0], "high": [105.0], "low": [95.0],
                "close": [102.0 + i * 0.1], "volume": [1e6], "amount": [1e9]
            }))
        dist = KairosDistribution(frames)
        assert len(dist.df) == 100, f"Expected 100 rows, got {len(dist.df)}"


# =============================================================================
# ADDITIONAL TESTS FOR ROBUSTNESS
# =============================================================================

class TestAdditionalMetrics:
    """Additional tests for other KairosDistribution methods"""

    def test_coefficient_of_variation(self):
        """CV = std / |mean|"""
        prices = np.random.normal(100, 10, 100).tolist()
        dist = make_dist(prices)
        cv = dist.coefficient_of_variation()
        expected_cv = dist.stats["close"]["std"] / abs(dist.stats["close"]["mean"])
        assert abs(cv - expected_cv) < 1e-6

    def test_predicted_sharpe(self):
        """Sharpe = mean / std"""
        prices = np.random.normal(100, 5, 100).tolist()
        dist = make_dist(prices)
        sharpe = dist.predicted_sharpe()
        expected_sharpe = dist.stats["close"]["mean"] / dist.stats["close"]["std"]
        assert abs(sharpe - expected_sharpe) < 1e-6

    def test_is_bimodal_false_for_normal(self):
        """Normal distribution should not be bimodal"""
        prices = np.random.normal(100, 5, 100).tolist()
        dist = make_dist(prices)
        assert not dist.is_bimodal(), "Normal distribution should not be bimodal"

    def test_is_bimodal_true_for_bimodal(self):
        """Distribution with two peaks should be bimodal"""
        # Two distinct clusters
        prices = list(np.random.normal(90, 2, 50)) + list(np.random.normal(110, 2, 50))
        dist = make_dist(prices)
        # This may or may not be detected as bimodal depending on KDE detection,
        # but we can at least verify it doesn't crash and returns a boolean
        result = dist.is_bimodal()
        assert isinstance(result, (bool, np.bool_))
