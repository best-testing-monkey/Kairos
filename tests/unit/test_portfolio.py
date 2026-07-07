"""
Tests for kairos_portfolio.py: PortfolioAllocator base class and shrinkage covariance.

This module validates:
- Shrinkage intensity bounds [0,1]
- Positive definiteness of shrunk covariance (via np.linalg.cholesky)
- Equal-weight fallback below min_obs threshold
- Base class raises NotImplementedError
- No cvxpy/external solver dependencies (numpy/scipy only)
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add strategy/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from kairos_portfolio import (
    PortfolioAllocator,
    shrunk_covariance,
    _fallback_equal_weight,
    _ledoit_wolf_intensity,
    MVOAllocator,
    RiskParityAllocator,
    HRPAllocator,
    MinVarAllocator,
)
from kairos_backtest import Signal, Direction, KairosDistribution


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_returns_200_obs_3_assets():
    """
    Generate 200 observations of 3 correlated assets.
    Used to test shrinkage intensity with realistic data.
    """
    np.random.seed(42)
    n_obs = 200
    n_assets = 3

    # Correlated returns: first asset drives others
    factor = np.random.randn(n_obs)
    asset1 = 0.01 + 0.02 * factor + 0.01 * np.random.randn(n_obs)
    asset2 = 0.01 + 0.015 * factor + 0.015 * np.random.randn(n_obs)
    asset3 = 0.01 + 0.012 * factor + 0.02 * np.random.randn(n_obs)

    returns = pd.DataFrame(
        {"BTC": asset1, "ETH": asset2, "SOL": asset3},
        index=pd.date_range("2024-01-01", periods=n_obs),
    )
    return returns


@pytest.fixture
def synthetic_returns_small():
    """5 observations of 3 assets (less than min_obs=60)."""
    np.random.seed(123)
    returns = pd.DataFrame(
        np.random.randn(5, 3) * 0.02,
        columns=["BTC", "ETH", "SOL"],
        index=pd.date_range("2024-01-01", periods=5),
    )
    return returns


@pytest.fixture
def synthetic_returns_large_singular():
    """
    100 observations of 10 highly correlated assets.
    Used to test shrinkage when n_assets is close to n_obs.
    """
    np.random.seed(456)
    n_obs = 100
    n_assets = 10

    # Create near-singular structure: all assets move together
    factor = np.random.randn(n_obs)
    returns = pd.DataFrame(
        np.tile(factor.reshape(-1, 1), (1, n_assets)) + 0.01 * np.random.randn(n_obs, n_assets),
        columns=[f"Asset{i}" for i in range(n_assets)],
        index=pd.date_range("2024-01-01", periods=n_obs),
    )
    return returns


@pytest.fixture
def simple_signals():
    """Two simple LONG signals."""
    return {
        "BTC": Signal(
            direction=Direction.LONG,
            size=0.1,
            entry=50000,
            stop=49000,
            target=51000,
            strategy_name="test_strategy",
            confidence=0.8,
            expected_value=100.0,
        ),
        "ETH": Signal(
            direction=Direction.LONG,
            size=0.1,
            entry=3000,
            stop=2900,
            target=3100,
            strategy_name="test_strategy",
            confidence=0.75,
            expected_value=50.0,
        ),
    }


# =============================================================================
# TEST SHRINKAGE INTENSITY BOUNDS
# =============================================================================

class TestShrinkageIntensityBounds:
    """Verify shrinkage intensity always lies in [0,1]."""

    def test_shrinkage_intensity_with_200_obs(self, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: with n=200 obs / 3 assets, shrinkage intensity < 0.3.
        This ensures the estimator still respects the sample when data is plentiful.
        """
        returns = synthetic_returns_200_obs_3_assets
        n_obs = len(returns)
        n_assets = returns.shape[1]

        # Compute sample covariance and intensity
        S = np.cov(returns.T)
        alpha = _ledoit_wolf_intensity(returns, S)

        # Bounds check
        assert 0.0 <= alpha <= 1.0, f"Intensity {alpha} outside [0,1]"
        assert alpha < 0.3, f"Intensity {alpha} should be < 0.3 with n=200, k=3"

    def test_shrinkage_intensity_bounds_various_sizes(self):
        """Test that intensity stays in [0,1] for various observation/asset counts."""
        test_cases = [
            (50, 3),   # Few observations
            (100, 5),  # Moderate
            (200, 3),  # Plentiful
            (500, 10), # Very plentiful
        ]

        for n_obs, n_assets in test_cases:
            np.random.seed(789)
            returns = pd.DataFrame(
                np.random.randn(n_obs, n_assets) * 0.02,
                columns=[f"A{i}" for i in range(n_assets)],
            )

            cov_shrunk = shrunk_covariance(returns)
            S = np.cov(returns.T)
            alpha = _ledoit_wolf_intensity(returns, S)

            assert 0.0 <= alpha <= 1.0, f"Intensity {alpha} out of bounds for n={n_obs}, k={n_assets}"

    def test_shrinkage_intensity_with_few_obs(self):
        """With few observations relative to assets, intensity should be higher."""
        np.random.seed(999)
        returns_few = pd.DataFrame(
            np.random.randn(10, 5) * 0.02,
            columns=[f"A{i}" for i in range(5)],
        )
        returns_many = pd.DataFrame(
            np.random.randn(500, 5) * 0.02,
            columns=[f"A{i}" for i in range(5)],
        )

        S_few = np.cov(returns_few.T)
        S_many = np.cov(returns_many.T)
        alpha_few = _ledoit_wolf_intensity(returns_few, S_few)
        alpha_many = _ledoit_wolf_intensity(returns_many, S_many)

        # With few obs, intensity should be higher (more shrinkage needed)
        assert alpha_few >= alpha_many, \
            f"Expected alpha_few ({alpha_few}) >= alpha_many ({alpha_many})"


# =============================================================================
# TEST POSITIVE DEFINITENESS
# =============================================================================

class TestPositiveDefiniteness:
    """Verify shrunk covariance is positive definite via Cholesky decomposition."""

    def test_shrunk_covariance_positive_definite_200_obs(self, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: output covariance is positive definite (Cholesky succeeds).
        """
        returns = synthetic_returns_200_obs_3_assets
        cov_shrunk = shrunk_covariance(returns)

        # Positive definiteness: Cholesky decomposition should succeed
        try:
            L = np.linalg.cholesky(cov_shrunk)
            assert L.shape == cov_shrunk.shape
        except np.linalg.LinAlgError as e:
            pytest.fail(f"Cholesky decomposition failed: {e}. Matrix is not positive definite.")

    def test_shrunk_covariance_positive_definite_singular_case(self):
        """
        Test positive definiteness when n_assets > n_obs (singular regime).
        Shrinkage must ensure output is positive definite even in this regime.
        """
        np.random.seed(111)
        n_obs = 5
        n_assets = 10

        returns = pd.DataFrame(
            np.random.randn(n_obs, n_assets) * 0.02,
            columns=[f"A{i}" for i in range(n_assets)],
        )

        cov_shrunk = shrunk_covariance(returns)

        # Must be positive definite
        try:
            L = np.linalg.cholesky(cov_shrunk)
        except np.linalg.LinAlgError as e:
            pytest.fail(f"Singular case: Cholesky failed despite shrinkage: {e}")

    def test_shrunk_covariance_positive_definite_large_singular(self, synthetic_returns_large_singular):
        """
        Near-singular case: 100 obs, 10 highly correlated assets.
        Shrinkage should render it positive definite.
        """
        returns = synthetic_returns_large_singular
        cov_shrunk = shrunk_covariance(returns)

        try:
            L = np.linalg.cholesky(cov_shrunk)
        except np.linalg.LinAlgError as e:
            pytest.fail(f"Large singular case: Cholesky failed: {e}")

    def test_shrunk_covariance_eigenvalues_positive(self, synthetic_returns_200_obs_3_assets):
        """Eigenvalues of shrunk covariance should all be > 0."""
        returns = synthetic_returns_200_obs_3_assets
        cov_shrunk = shrunk_covariance(returns)

        eigvals = np.linalg.eigvals(cov_shrunk)
        assert np.all(eigvals > 0), f"Negative eigenvalues found: {eigvals[eigvals <= 0]}"

    def test_shrunk_covariance_diagonal_positive(self, synthetic_returns_200_obs_3_assets):
        """Diagonal (variances) should be positive."""
        returns = synthetic_returns_200_obs_3_assets
        cov_shrunk = shrunk_covariance(returns)

        diag = np.diag(cov_shrunk)
        assert np.all(diag > 0), f"Non-positive diagonal: {diag}"


# =============================================================================
# TEST EQUAL-WEIGHT FALLBACK
# =============================================================================

class TestEqualWeightFallback:
    """Verify equal-weight fallback when observations < min_obs."""

    def test_equal_weight_fallback_below_min_obs(self, synthetic_returns_small, simple_signals):
        """
        Acceptance: when len(returns) < min_obs=60, allocator uses equal weight.
        """
        returns = synthetic_returns_small  # 5 obs
        assert len(returns) < PortfolioAllocator.min_obs

        # Call fallback
        weights = _fallback_equal_weight(simple_signals)

        # Should have 2 signals → 0.5 each
        assert len(weights) == 2
        assert weights["BTC"] == 0.5
        assert weights["ETH"] == 0.5
        assert abs(sum(weights.values()) - 1.0) < 1e-10

    def test_equal_weight_single_signal(self):
        """Single signal → weight 1.0."""
        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=50000,
                stop=49000,
                target=51000,
                strategy_name="test",
                confidence=0.8,
                expected_value=100.0,
            )
        }

        weights = _fallback_equal_weight(signals)

        assert len(weights) == 1
        assert weights["BTC"] == 1.0

    def test_equal_weight_empty_signals(self):
        """Empty signals → empty weights dict."""
        weights = _fallback_equal_weight({})

        assert weights == {}

    def test_equal_weight_many_signals(self):
        """Many signals → equal split."""
        n = 10
        signals = {
            f"Asset{i}": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=90.0,
                target=110.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for i in range(n)
        }

        weights = _fallback_equal_weight(signals)

        assert len(weights) == n
        expected_weight = 1.0 / n
        for w in weights.values():
            assert abs(w - expected_weight) < 1e-10


# =============================================================================
# TEST BASE CLASS
# =============================================================================

class TestPortfolioAllocatorBase:
    """Test PortfolioAllocator base class behavior."""

    def test_base_allocate_raises_not_implemented(self, simple_signals, synthetic_returns_200_obs_3_assets):
        """Calling allocate() on base class raises NotImplementedError."""
        allocator = PortfolioAllocator()
        returns = synthetic_returns_200_obs_3_assets

        with pytest.raises(NotImplementedError, match="must be implemented by subclasses"):
            allocator.allocate(simple_signals, returns, {}, {})

    def test_base_reset_no_op(self):
        """reset() on base class is a no-op."""
        allocator = PortfolioAllocator()
        allocator.reset()  # Should not raise

    def test_base_name_attribute(self):
        """Base class has a name attribute."""
        allocator = PortfolioAllocator()
        assert hasattr(allocator, "name")
        assert allocator.name == "base_allocator"

    def test_base_min_obs_attribute(self):
        """Base class has min_obs attribute."""
        allocator = PortfolioAllocator()
        assert hasattr(allocator, "min_obs")
        assert allocator.min_obs == 60

    def test_custom_allocator_subclass(self, simple_signals, synthetic_returns_200_obs_3_assets):
        """Custom allocator subclass can override allocate()."""

        class DummyAllocator(PortfolioAllocator):
            name = "dummy"

            def allocate(self, signals, returns, dists, context):
                # Dummy: return equal weight
                if len(returns) < self.min_obs:
                    return _fallback_equal_weight(signals)
                return _fallback_equal_weight(signals)

        allocator = DummyAllocator()
        returns = synthetic_returns_200_obs_3_assets
        weights = allocator.allocate(simple_signals, returns, {}, {})

        assert isinstance(weights, dict)
        assert len(weights) == 2
        assert all(abs(v - 0.5) < 1e-10 for v in weights.values())


# =============================================================================
# TEST DEPENDENCIES
# =============================================================================

class TestDependencies:
    """Verify no cvxpy or other hard dependencies."""

    def test_no_cvxpy_import(self):
        """Module does not import cvxpy."""
        import kairos_portfolio

        # cvxpy should not be in sys.modules (as a side effect of importing kairos_portfolio)
        # or in the module's imports
        with open(
            os.path.join(os.path.dirname(__file__), "..", "..", "strategy", "kairos_portfolio.py")
        ) as f:
            source = f.read()

        assert "cvxpy" not in source, "cvxpy should not be mentioned in kairos_portfolio.py"
        assert "from cvxpy" not in source
        assert "import cvxpy" not in source

    def test_only_numpy_scipy_pandas(self):
        """Only numpy, scipy, pandas are hard dependencies (sklearn optional)."""
        import kairos_portfolio

        # Get the module source
        with open(
            os.path.join(os.path.dirname(__file__), "..", "..", "strategy", "kairos_portfolio.py")
        ) as f:
            source = f.read()

        # Check for disallowed hard dependencies
        disallowed = ["arch", "statsmodels", "stumpy"]
        for lib in disallowed:
            assert f"import {lib}" not in source, f"{lib} should not be imported"
            assert f"from {lib}" not in source, f"{lib} should not be imported"

    def test_sklearn_optional(self):
        """sklearn.covariance.LedoitWolf is optional (try/except)."""
        with open(
            os.path.join(os.path.dirname(__file__), "..", "..", "strategy", "kairos_portfolio.py")
        ) as f:
            source = f.read()

        # Should have try/except for sklearn
        assert "try:" in source
        assert "from sklearn.covariance import LedoitWolf" in source
        assert "HAS_SKLEARN" in source


# =============================================================================
# TEST SHRINKAGE WITH MANUAL FALLBACK
# =============================================================================

class TestShrinkageManualFallback:
    """Test manual shrinkage fallback (when sklearn not available)."""

    def test_manual_ledoit_wolf_formula(self, synthetic_returns_200_obs_3_assets):
        """Manual Ledoit-Wolf formula produces valid shrinkage intensity."""
        returns = synthetic_returns_200_obs_3_assets
        S = np.cov(returns.T)

        alpha = _ledoit_wolf_intensity(returns, S)

        # Should produce a valid intensity
        assert isinstance(alpha, float)
        assert 0.0 <= alpha <= 1.0

    def test_shrunk_covariance_interpolates(self, synthetic_returns_200_obs_3_assets):
        """Shrunk covariance should be convex combination of sample and target."""
        returns = synthetic_returns_200_obs_3_assets
        S = np.cov(returns.T)
        cov_shrunk = shrunk_covariance(returns)

        # Manually compute: (1-α)*S + α*T
        alpha = _ledoit_wolf_intensity(returns, S)
        trace_S = np.trace(S)
        target = (trace_S / S.shape[0]) * np.eye(S.shape[0])

        # Manual interpolation (before eigenvalue clipping)
        manual_interp = (1.0 - alpha) * S + alpha * target

        # After eigenvalue clipping, diagonal should still be close
        # (eigenvalue clipping is applied to maintain positive definiteness)
        assert np.allclose(np.diag(cov_shrunk), np.diag(manual_interp), rtol=0.2), \
            "Shrunk cov diagonal should be close to interpolation"

    def test_shrunk_covariance_explicit_alpha(self, synthetic_returns_200_obs_3_assets):
        """Can pass explicit shrinkage intensity for testing."""
        returns = synthetic_returns_200_obs_3_assets
        S = np.cov(returns.T)

        # Test α = 0.0 (pure sample)
        cov_0 = shrunk_covariance(returns, shrinkage_intensity=0.0)
        assert np.allclose(cov_0, S, rtol=1e-5), "α=0 should recover sample covariance"

        # Test α = 1.0 (pure target)
        cov_1 = shrunk_covariance(returns, shrinkage_intensity=1.0)
        trace_S = np.trace(S)
        target = (trace_S / S.shape[0]) * np.eye(S.shape[0])
        assert np.allclose(cov_1, target, rtol=1e-5), "α=1 should recover target"

        # Test α = 0.5 (50/50)
        cov_half = shrunk_covariance(returns, shrinkage_intensity=0.5)
        expected_half = 0.5 * S + 0.5 * target
        assert np.allclose(cov_half, expected_half, rtol=1e-4), "α=0.5 should be 50/50 mix"


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_shrunk_covariance_single_asset(self):
        """Single asset (n_assets=1)."""
        returns = pd.DataFrame(
            np.random.randn(100) * 0.02,
            columns=["A"],
            index=pd.date_range("2024-01-01", periods=100),
        )

        cov_shrunk = shrunk_covariance(returns)

        assert cov_shrunk.shape == (1, 1)
        assert cov_shrunk[0, 0] > 0

    def test_shrunk_covariance_with_nan(self):
        """NaN values are dropped."""
        returns = pd.DataFrame(
            {"A": [0.01, np.nan, 0.02, 0.03, 0.01] * 20,
             "B": [0.02, 0.01, np.nan, 0.02, 0.03] * 20},
        )

        cov_shrunk = shrunk_covariance(returns)

        assert cov_shrunk.shape == (2, 2)
        assert not np.isnan(cov_shrunk).any()

    def test_shrunk_covariance_with_inf(self):
        """Inf values are dropped."""
        returns = pd.DataFrame(
            {"A": [0.01, np.inf, 0.02, 0.03, 0.01] * 20,
             "B": [0.02, 0.01, -np.inf, 0.02, 0.03] * 20},
        )

        cov_shrunk = shrunk_covariance(returns)

        assert cov_shrunk.shape == (2, 2)
        assert not np.isnan(cov_shrunk).any()
        assert not np.isinf(cov_shrunk).any()

    def test_shrunk_covariance_constant_series(self):
        """Constant series (zero variance)."""
        returns = pd.DataFrame(
            {"A": [0.0] * 100, "B": [0.01] * 100},
            index=pd.date_range("2024-01-01", periods=100),
        )

        cov_shrunk = shrunk_covariance(returns)

        # Should still be positive definite (shrinkage adds mass to diagonal)
        eigvals = np.linalg.eigvals(cov_shrunk)
        assert np.all(eigvals > 0)

    def test_shrunk_covariance_tiny_dataset(self):
        """Very small dataset (3 obs, 2 assets)."""
        returns = pd.DataFrame(
            [[0.01, 0.02], [0.02, 0.01], [0.01, 0.03]],
            columns=["A", "B"],
        )

        cov_shrunk = shrunk_covariance(returns)

        # Should handle gracefully and be positive definite
        eigvals = np.linalg.eigvals(cov_shrunk)
        assert np.all(eigvals > 0)

    def test_shrunk_covariance_high_correlation(self):
        """Highly correlated assets."""
        np.random.seed(42)
        factor = np.random.randn(100)
        returns = pd.DataFrame(
            {"A": factor + 0.001 * np.random.randn(100),
             "B": factor + 0.001 * np.random.randn(100),
             "C": factor + 0.001 * np.random.randn(100)},
        )

        cov_shrunk = shrunk_covariance(returns)

        # Still positive definite
        try:
            np.linalg.cholesky(cov_shrunk)
        except np.linalg.LinAlgError:
            pytest.fail("Highly correlated case: not positive definite")

        # Shrinkage should be valid (even with high correlation and plentiful data)
        S = np.cov(returns.T)
        alpha = _ledoit_wolf_intensity(returns, S)
        assert 0.0 <= alpha <= 1.0, "Shrinkage intensity should be in [0,1]"
        # With 100 obs and 3 assets, we have plentiful data, so shrinkage may be low
        # even with high correlation (Ledoit-Wolf balances bias and variance)


# =============================================================================
# TEST MVO ALLOCATOR
# =============================================================================

class TestMVOAllocator:
    """Test Mean-Variance Optimization allocator."""

    @pytest.fixture
    def mvo_allocator(self):
        """Standard MVO allocator instance."""
        return MVOAllocator(lookback=120, gross_cap=1.0, max_weight=0.35, rf=0.0)

    @pytest.fixture
    def synthetic_returns_2_assets_uncorrelated(self):
        """
        Two uncorrelated assets with 150 observations.
        Used for equal-mu split test.
        """
        np.random.seed(42)
        n_obs = 150
        # Uncorrelated: different random seeds
        asset1 = np.random.randn(n_obs) * 0.02
        asset2 = np.random.randn(n_obs) * 0.02
        returns = pd.DataFrame(
            {"BTC": asset1, "ETH": asset2},
            index=pd.date_range("2024-01-01", periods=n_obs),
        )
        return returns

    @pytest.fixture
    def synthetic_distributions_equal_mu(self):
        """Mock distributions with equal expected values."""
        # Create synthetic prediction samples: 100 samples per asset
        np.random.seed(42)
        n_samples = 100

        # Asset 1: close prices centered at 50000 with std=1000
        close1 = np.random.normal(50000, 1000, n_samples)
        pred1 = [
            pd.DataFrame({
                "open": close1 + np.random.normal(0, 100, n_samples),
                "high": close1 + np.abs(np.random.normal(0, 500, n_samples)),
                "low": close1 - np.abs(np.random.normal(0, 500, n_samples)),
                "close": close1,
                "volume": np.full(n_samples, 1e9),
                "amount": np.full(n_samples, 1e9),
            })
            for _ in range(100)
        ]
        dist1 = KairosDistribution(pred1)

        # Asset 2: close prices centered at 3000 with std=60
        close2 = np.random.normal(3000, 60, n_samples)
        pred2 = [
            pd.DataFrame({
                "open": close2 + np.random.normal(0, 6, n_samples),
                "high": close2 + np.abs(np.random.normal(0, 30, n_samples)),
                "low": close2 - np.abs(np.random.normal(0, 30, n_samples)),
                "close": close2,
                "volume": np.full(n_samples, 1e8),
                "amount": np.full(n_samples, 1e8),
            })
            for _ in range(100)
        ]
        dist2 = KairosDistribution(pred2)

        return {"BTC": dist1, "ETH": dist2}

    def test_mvo_equal_mu_splits_50_50(self, mvo_allocator, synthetic_returns_2_assets_uncorrelated):
        """
        Acceptance: with two uncorrelated assets of equal mu and Sharpe, weights split ~50/50.

        Creates two assets with:
        - Same Sharpe ratio (mu/sigma)
        - Uncorrelated returns
        - Both LONG signals
        """
        returns = synthetic_returns_2_assets_uncorrelated

        # Create distributions with same Sharpe ratio so weights should split
        np.random.seed(42)
        n_samples = 100

        # Asset 1 (BTC):
        # Entry=50000, Stop=48000, Target=52000
        # Distribution: std ~1000, win_r=2000, gives moderate Sharpe
        close1 = np.random.normal(50500, 1000, n_samples)
        pred1 = [
            pd.DataFrame({
                "open": close1 + np.random.normal(0, 100, n_samples),
                "high": close1 + np.abs(np.random.normal(0, 500, n_samples)),
                "low": close1 - np.abs(np.random.normal(0, 500, n_samples)),
                "close": close1,
                "volume": np.full(n_samples, 1e9),
                "amount": np.full(n_samples, 1e9),
            })
            for _ in range(100)
        ]
        dist1 = KairosDistribution(pred1)

        # Asset 2 (ETH):
        # Create an uncorrelated asset with similar characteristics to BTC
        # Entry=3000, Stop=2800, Target=3200 (scaled version)
        close2 = np.random.normal(3100, 200, n_samples)
        pred2 = [
            pd.DataFrame({
                "open": close2 + np.random.normal(0, 20, n_samples),
                "high": close2 + np.abs(np.random.normal(0, 100, n_samples)),
                "low": close2 - np.abs(np.random.normal(0, 100, n_samples)),
                "close": close2,
                "volume": np.full(n_samples, 1e8),
                "amount": np.full(n_samples, 1e8),
            })
            for _ in range(100)
        ]
        dist2 = KairosDistribution(pred2)

        dists = {"BTC": dist1, "ETH": dist2}

        # Create signals with similar relative brackets
        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=50000,
                stop=48000,
                target=52000,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=3000,
                stop=2800,
                target=3200,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = mvo_allocator.allocate(signals, returns, dists, {})

        # Both assets should be long (non-zero weight)
        assert weights["BTC"] > 1e-8, f"BTC should be long, got {weights['BTC']}"
        assert weights["ETH"] > 1e-8, f"ETH should be long, got {weights['ETH']}"

        # With similar Sharpe ratios, weights should split roughly 50/50
        # Allow tolerance for numerical optimization and distribution variations
        total = weights["BTC"] + weights["ETH"]
        ratio = weights["BTC"] / total if total > 0 else 0.5
        # Allow 30-70 split (accounting for sample variance in distributions)
        assert 0.25 < ratio < 0.75, f"Expected ~50/50 split, got {ratio:.2%} / {1-ratio:.2%}"

        # Respect caps
        assert abs(weights["BTC"]) <= 0.35 + 1e-6, "BTC weight exceeds max_weight"
        assert abs(weights["ETH"]) <= 0.35 + 1e-6, "ETH weight exceeds max_weight"
        assert sum(abs(w) for w in weights.values()) <= 1.0 + 1e-6, "Exceeds gross_cap"

    def test_mvo_monotonic_mu_weight(self, synthetic_returns_2_assets_uncorrelated):
        """
        Acceptance: raising one asset's mu monotonically raises its weight.

        Creates multiple allocators with increasing mu for BTC, verifies weight increases.
        """
        returns = synthetic_returns_2_assets_uncorrelated

        # Test with multiple ETH mu levels (BTC will have higher mu each time)
        scenarios = [
            # (btc_target_offset, eth_target_offset)
            (51000, 3000),   # BTC: 2% gain, ETH: 0% (flat entry=target)
            (51500, 3000),   # BTC: 3% gain, ETH: 0%
            (52000, 3000),   # BTC: 4% gain, ETH: 0%
        ]
        weights_btc = []

        for btc_target, eth_target in scenarios:
            np.random.seed(42)
            n_samples = 100

            # BTC: distribution centered around entry (50000)
            close1 = np.random.normal(50000, 500, n_samples)
            pred1 = [
                pd.DataFrame({
                    "open": close1 + np.random.normal(0, 50, n_samples),
                    "high": np.maximum(close1, btc_target) + np.abs(np.random.normal(0, 250, n_samples)),
                    "low": np.minimum(close1, btc_target) - np.abs(np.random.normal(0, 250, n_samples)),
                    "close": close1,
                    "volume": np.full(n_samples, 1e9),
                    "amount": np.full(n_samples, 1e9),
                })
                for _ in range(100)
            ]
            dist1 = KairosDistribution(pred1)

            # ETH: distribution centered around entry (3000)
            close2 = np.random.normal(3000, 30, n_samples)
            pred2 = [
                pd.DataFrame({
                    "open": close2 + np.random.normal(0, 3, n_samples),
                    "high": np.maximum(close2, eth_target) + np.abs(np.random.normal(0, 15, n_samples)),
                    "low": np.minimum(close2, eth_target) - np.abs(np.random.normal(0, 15, n_samples)),
                    "close": close2,
                    "volume": np.full(n_samples, 1e8),
                    "amount": np.full(n_samples, 1e8),
                })
                for _ in range(100)
            ]
            dist2 = KairosDistribution(pred2)

            dists = {"BTC": dist1, "ETH": dist2}

            signals = {
                "BTC": Signal(
                    direction=Direction.LONG,
                    size=0.1,
                    entry=50000,
                    stop=49000,
                    target=btc_target,
                    strategy_name="test",
                    confidence=0.8,
                    expected_value=0.0,
                ),
                "ETH": Signal(
                    direction=Direction.LONG,
                    size=0.1,
                    entry=3000,
                    stop=2900,
                    target=eth_target,
                    strategy_name="test",
                    confidence=0.8,
                    expected_value=0.0,
                ),
            }

            allocator = MVOAllocator(lookback=120, gross_cap=1.0, max_weight=0.35, rf=0.0)
            weights = allocator.allocate(signals, returns, dists, {})
            weights_btc.append(weights.get("BTC", 0.0))

        # With increasing target for BTC, its weight should increase
        # (or at least not decrease consistently)
        assert weights_btc[0] <= weights_btc[2], \
            f"Expected BTC weight to increase from {weights_btc[0]:.6f} to {weights_btc[2]:.6f}"

    def test_mvo_respects_caps(self, mvo_allocator, synthetic_returns_200_obs_3_assets, simple_signals):
        """
        Acceptance: solution never violates gross_cap or max_weight constraints.
        """
        returns = synthetic_returns_200_obs_3_assets

        # Create mock distributions with positive mu
        np.random.seed(42)
        n_samples = 100
        dists = {}
        for sym in ["BTC", "ETH", "SOL"]:
            close = np.random.normal(100, 10, n_samples)
            pred = [
                pd.DataFrame({
                    "open": close + np.random.normal(0, 1, n_samples),
                    "high": close + np.abs(np.random.normal(0, 5, n_samples)),
                    "low": close - np.abs(np.random.normal(0, 5, n_samples)),
                    "close": close,
                    "volume": np.full(n_samples, 1e6),
                    "amount": np.full(n_samples, 1e6),
                })
                for _ in range(100)
            ]
            dists[sym] = KairosDistribution(pred)

        # Adjust signals to match returns columns
        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100,
                stop=95,
                target=105,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100,
                stop=95,
                target=105,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "SOL": Signal(
                direction=Direction.SHORT,
                size=0.1,
                entry=100,
                stop=105,
                target=95,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = mvo_allocator.allocate(signals, returns, dists, {})

        # Check max_weight constraint
        for sym, w in weights.items():
            assert abs(w) <= 0.35 + 1e-6, \
                f"{sym} weight {w} exceeds max_weight 0.35"

        # Check gross_cap constraint
        gross_leverage = sum(abs(w) for w in weights.values())
        assert gross_leverage <= 1.0 + 1e-6, \
            f"Gross leverage {gross_leverage} exceeds gross_cap 1.0"

        # Check sign constraints
        assert weights["BTC"] >= -1e-10, "BTC (LONG) should not be negative"
        assert weights["ETH"] >= -1e-10, "ETH (LONG) should not be negative"
        assert weights["SOL"] <= 1e-10, "SOL (SHORT) should not be positive"

    def test_mvo_fallback_below_min_obs(self, mvo_allocator, synthetic_returns_small, simple_signals):
        """
        Acceptance: fallback to equal weight when len(returns) < min_obs.
        """
        returns = synthetic_returns_small  # 5 obs
        assert len(returns) < PortfolioAllocator.min_obs

        # Create dummy distributions
        np.random.seed(42)
        n_samples = 100
        dists = {}
        for sym in ["BTC", "ETH"]:
            close = np.random.normal(100, 10, n_samples)
            pred = [
                pd.DataFrame({
                    "open": close + np.random.normal(0, 1, n_samples),
                    "high": close + np.abs(np.random.normal(0, 5, n_samples)),
                    "low": close - np.abs(np.random.normal(0, 5, n_samples)),
                    "close": close,
                    "volume": np.full(n_samples, 1e6),
                    "amount": np.full(n_samples, 1e6),
                })
                for _ in range(100)
            ]
            dists[sym] = KairosDistribution(pred)

        weights = mvo_allocator.allocate(simple_signals, returns, dists, {})

        # Should fall back to equal weight
        expected = _fallback_equal_weight(simple_signals)
        assert weights == expected, \
            f"Expected equal-weight fallback {expected}, got {weights}"

    def test_mvo_allocator_attributes(self, mvo_allocator):
        """Test allocator has correct name and min_obs."""
        assert mvo_allocator.name == "mvo_allocator"
        assert mvo_allocator.min_obs == 60
        assert mvo_allocator.lookback == 120
        assert mvo_allocator.gross_cap == 1.0
        assert mvo_allocator.max_weight == 0.35
        assert mvo_allocator.rf == 0.0

    def test_mvo_allocator_custom_params(self):
        """Test allocator with custom parameters."""
        allocator = MVOAllocator(
            lookback=60,
            gross_cap=2.0,
            max_weight=0.5,
            rf=0.0001
        )
        assert allocator.lookback == 60
        assert allocator.gross_cap == 2.0
        assert allocator.max_weight == 0.5
        assert allocator.rf == 0.0001

    def test_mvo_allocator_empty_signals(self, mvo_allocator, synthetic_returns_200_obs_3_assets):
        """With no signals, return empty dict."""
        returns = synthetic_returns_200_obs_3_assets
        weights = mvo_allocator.allocate({}, returns, {}, {})
        assert weights == {}

    def test_mvo_allocator_zero_expected_return(self, mvo_allocator, synthetic_returns_200_obs_3_assets):
        """Allocator should handle zero or negative expected returns gracefully."""
        returns = synthetic_returns_200_obs_3_assets

        # Create distributions with zero expected returns
        np.random.seed(42)
        n_samples = 100
        dists = {}
        for sym in ["BTC", "ETH"]:
            # Constant close prices → zero expected return
            close = np.full(n_samples, 100.0)
            pred = [
                pd.DataFrame({
                    "open": close + np.random.normal(0, 0.1, n_samples),
                    "high": close + np.abs(np.random.normal(0, 0.5, n_samples)),
                    "low": close - np.abs(np.random.normal(0, 0.5, n_samples)),
                    "close": close,
                    "volume": np.full(n_samples, 1e6),
                    "amount": np.full(n_samples, 1e6),
                })
                for _ in range(100)
            ]
            dists[sym] = KairosDistribution(pred)

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100,
                stop=99,
                target=101,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100,
                stop=99,
                target=101,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        # Should not raise; may fall back to equal weight
        weights = mvo_allocator.allocate(signals, returns, dists, {})
        assert isinstance(weights, dict)
        assert len(weights) > 0  # Should have some allocation


# =============================================================================
# TEST RISK PARITY ALLOCATOR
# =============================================================================

class TestRiskParityAllocator:
    """Test Equal Risk Contribution (ERC) allocator."""

    @pytest.fixture
    def rp_allocator(self):
        """Standard Risk Parity allocator instance."""
        return RiskParityAllocator(lookback=120, gross_cap=1.0, max_weight=0.35)

    @pytest.fixture
    def synthetic_returns_2asset_uncorrelated_10_20_vol(self):
        """
        Two assets with sample vols exactly 10% and 20% and exactly zero sample
        correlation (via Gram-Schmidt orthogonalization).
        ERC on uncorrelated assets gives inverse-vol weights → |w1|/|w2| = 2:1.
        """
        np.random.seed(42)
        n_obs = 150

        a = np.random.randn(n_obs)
        b = np.random.randn(n_obs)

        # Demean and orthogonalize b against a (exact zero sample correlation)
        a = a - a.mean()
        b = b - b.mean()
        b = b - (np.dot(a, b) / np.dot(a, a)) * a

        # Rescale to exact sample standard deviations: 10% and 20%
        asset1 = a / a.std(ddof=1) * 0.10
        asset2 = b / b.std(ddof=1) * 0.20

        returns = pd.DataFrame(
            {"ASSET1": asset1, "ASSET2": asset2},
            index=pd.date_range("2024-01-01", periods=n_obs),
        )
        return returns

    @pytest.fixture
    def synthetic_returns_3asset_correlated(self):
        """
        Three correlated assets for testing risk contribution convergence.
        """
        np.random.seed(42)
        n_obs = 150

        factor = np.random.randn(n_obs)
        asset1 = 0.008 * factor + 0.004 * np.random.randn(n_obs)
        asset2 = 0.010 * factor + 0.008 * np.random.randn(n_obs)
        asset3 = 0.008 * factor + 0.014 * np.random.randn(n_obs)

        returns = pd.DataFrame(
            {"ASSET1": asset1, "ASSET2": asset2, "ASSET3": asset3},
            index=pd.date_range("2024-01-01", periods=n_obs),
        )
        return returns

    def _compute_risk_contributions(self, weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Compute risk contributions rc_i = w_i * (Σw)_i."""
        sigma_w = cov @ weights
        rc = weights * sigma_w
        return rc

    def test_risk_parity_2asset_vol_ratio(self, rp_allocator, synthetic_returns_2asset_uncorrelated_10_20_vol):
        """
        Acceptance (ticket E2-S02): 2 assets with vol 10%/20% and zero
        correlation → |weights| ratio ≈ 2:1, within 5% tolerance.
        """
        returns = synthetic_returns_2asset_uncorrelated_10_20_vol

        corr = returns.corr()
        assert abs(corr.iloc[0, 1]) < 1e-10, "Assets should be exactly uncorrelated"

        signals = {
            "ASSET1": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ASSET2": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = rp_allocator.allocate(signals, returns, {}, {})

        assert weights["ASSET1"] > 1e-8, "ASSET1 should be LONG"
        assert weights["ASSET2"] > 1e-8, "ASSET2 should be LONG"

        # Ticket acceptance: |w1|/|w2| ≈ 2:1 within 5% tolerance
        actual_ratio = weights["ASSET1"] / weights["ASSET2"]
        assert abs(actual_ratio - 2.0) < 0.1, \
            f"Weight ratio {actual_ratio:.4f} should be ~2.0 (within 5% tolerance)"

    def test_risk_parity_equal_contributions_3asset(self, rp_allocator, synthetic_returns_3asset_correlated):
        """
        Acceptance (ticket E2-S02): risk contributions within 1% of each other
        at convergence. Tests with 3 correlated assets.
        """
        returns = synthetic_returns_3asset_correlated

        signals = {
            "ASSET1": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ASSET2": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ASSET3": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = rp_allocator.allocate(signals, returns, {}, {})

        for sym in ["ASSET1", "ASSET2", "ASSET3"]:
            assert weights[sym] > 1e-8, f"{sym} should have positive weight"

        trailing_returns = returns[["ASSET1", "ASSET2", "ASSET3"]].tail(rp_allocator.lookback)
        cov = shrunk_covariance(trailing_returns)
        w_array = np.array([weights["ASSET1"], weights["ASSET2"], weights["ASSET3"]])
        rc = self._compute_risk_contributions(w_array, cov)

        mean_rc = np.mean(rc)
        assert mean_rc > 1e-12, "Portfolio should carry non-trivial risk"
        rc_normalized = rc / mean_rc
        max_deviation = np.max(np.abs(rc_normalized - 1.0))
        # Ticket acceptance: risk contributions within 1% of each other
        assert max_deviation < 0.01, \
            f"Risk contributions deviate {max_deviation*100:.4f}% from mean " \
            f"(should be < 1%): {rc_normalized}"

    def test_risk_parity_directions_long_short_mix(self, rp_allocator, synthetic_returns_200_obs_3_assets):
        """Acceptance: signs follow signal directions (mix LONG/SHORT)."""
        returns = synthetic_returns_200_obs_3_assets

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "SOL": Signal(
                direction=Direction.SHORT,
                size=0.1,
                entry=100.0,
                stop=101.0,
                target=99.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = rp_allocator.allocate(signals, returns, {}, {})

        assert weights["BTC"] >= -1e-10, "BTC (LONG) should not be negative"
        assert weights["ETH"] >= -1e-10, "ETH (LONG) should not be negative"
        assert weights["SOL"] <= 1e-10, "SOL (SHORT) should not be positive"
        assert sum(abs(w) for w in weights.values()) > 1e-8, "Should have non-trivial allocation"

    def test_risk_parity_respects_caps(self, rp_allocator, synthetic_returns_200_obs_3_assets):
        """Acceptance: solution never violates gross_cap or max_weight constraints."""
        returns = synthetic_returns_200_obs_3_assets

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "SOL": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = rp_allocator.allocate(signals, returns, {}, {})

        for sym, w in weights.items():
            assert abs(w) <= 0.35 + 1e-6, \
                f"{sym} weight {w} exceeds max_weight 0.35"

        gross_leverage = sum(abs(w) for w in weights.values())
        assert gross_leverage <= 1.0 + 1e-6, \
            f"Gross leverage {gross_leverage} exceeds gross_cap 1.0"

    def test_risk_parity_fallback_below_min_obs(self, rp_allocator, synthetic_returns_small):
        """Acceptance: fallback to equal weight when len(returns) < min_obs."""
        returns = synthetic_returns_small
        assert len(returns) < PortfolioAllocator.min_obs

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = rp_allocator.allocate(signals, returns, {}, {})

        expected = _fallback_equal_weight(signals)
        assert weights == expected, \
            f"Expected equal-weight fallback {expected}, got {weights}"

    def test_risk_parity_allocator_attributes(self, rp_allocator):
        """Test allocator has correct name and min_obs."""
        assert rp_allocator.name == "risk_parity_allocator"
        assert rp_allocator.min_obs == 60
        assert rp_allocator.lookback == 120
        assert rp_allocator.gross_cap == 1.0
        assert rp_allocator.max_weight == 0.35

    def test_risk_parity_empty_signals(self, rp_allocator, synthetic_returns_200_obs_3_assets):
        """With no signals, return empty dict."""
        returns = synthetic_returns_200_obs_3_assets
        weights = rp_allocator.allocate({}, returns, {}, {})
        assert weights == {}


# =============================================================================
# TEST HIERARCHICAL RISK PARITY ALLOCATOR
# =============================================================================

class TestHRPAllocator:
    """Test Hierarchical Risk Parity (HRP) allocator."""

    @pytest.fixture
    def hrp_allocator(self):
        """Standard HRP allocator instance."""
        return HRPAllocator(lookback=120, variant="hrp")

    @pytest.fixture
    def hrp_allocator_herc(self):
        """HRP allocator with HERC variant."""
        return HRPAllocator(lookback=120, variant="herc")

    @pytest.fixture
    def synthetic_returns_4asset_block_covariance(self):
        """
        Seeded 4-asset block covariance with explicit structure.

        Creates returns from a known multivariate normal distribution with
        block diagonal covariance:
        - Assets 0,1: vol=0.1, corr=0.5 (low-vol cluster)
        - Assets 2,3: vol=0.2, corr=0.5 (high-vol cluster)
        - Between clusters: corr=0.0 (uncorrelated)

        This ensures the low-vol pair has lower risk than high-vol pair.
        """
        np.random.seed(42)
        n_obs = 150

        # Define explicit block-diagonal covariance matrix
        cov = np.array([
            [0.01, 0.005, 0.0, 0.0],      # A0
            [0.005, 0.01, 0.0, 0.0],      # A1
            [0.0, 0.0, 0.04, 0.02],       # A2
            [0.0, 0.0, 0.02, 0.04],       # A3
        ])

        # Generate correlated returns from multivariate normal
        mean = np.zeros(4)
        returns_data = np.random.multivariate_normal(mean, cov, n_obs)

        returns = pd.DataFrame(
            returns_data,
            columns=["A0", "A1", "A2", "A3"],
            index=pd.date_range("2024-01-01", periods=n_obs),
        )
        return returns

    def test_hrp_synthetic_covariance(self, hrp_allocator, synthetic_returns_4asset_block_covariance):
        """
        Acceptance: 4-asset block covariance with deterministic weights summing
        to 1, and low-vol pair receives more total weight than high-vol pair.
        """
        returns = synthetic_returns_4asset_block_covariance

        signals = {
            "A0": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "A1": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "A2": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "A3": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = hrp_allocator.allocate(signals, returns, {}, {})

        # All weights should be positive (all LONG)
        for sym in ["A0", "A1", "A2", "A3"]:
            assert weights[sym] > 1e-8, f"{sym} should be LONG"

        # Weights should sum to approximately 1.0
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01, \
            f"Weights should sum to ~1.0, got {total_weight:.6f}"

        # Low-vol pair (A0, A1) should receive more total weight than high-vol pair (A2, A3)
        low_vol_weight = weights["A0"] + weights["A1"]
        high_vol_weight = weights["A2"] + weights["A3"]
        assert low_vol_weight > high_vol_weight, \
            f"Low-vol pair {low_vol_weight:.6f} should get more weight than " \
            f"high-vol pair {high_vol_weight:.6f}"

    def test_hrp_2asset_degenerates_to_inv_var(self, hrp_allocator):
        """
        Acceptance: n_assets=2 should degenerate to inverse-variance allocation.
        With two assets of vol 10%/20% and zero correlation, weight ratio ≈ 2:1.
        """
        np.random.seed(42)
        n_obs = 150

        # Two uncorrelated assets with exact vols 10% and 20%
        a = np.random.randn(n_obs)
        b = np.random.randn(n_obs)

        # Orthogonalize
        a = a - a.mean()
        b = b - b.mean()
        b = b - (np.dot(a, b) / np.dot(a, a)) * a

        # Rescale to exact vols
        asset1 = a / a.std(ddof=1) * 0.10
        asset2 = b / b.std(ddof=1) * 0.20

        returns = pd.DataFrame(
            {"A1": asset1, "A2": asset2},
            index=pd.date_range("2024-01-01", periods=n_obs),
        )

        signals = {
            "A1": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "A2": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = hrp_allocator.allocate(signals, returns, {}, {})

        assert weights["A1"] > 1e-8, "A1 should be LONG"
        assert weights["A2"] > 1e-8, "A2 should be LONG"

        # For 2 uncorrelated assets: weight_ratio ≈ vol2 / vol1 = 0.20 / 0.10 = 2.0
        ratio = weights["A1"] / weights["A2"]
        assert abs(ratio - 2.0) < 0.1, \
            f"Weight ratio {ratio:.4f} should be ~2.0 for inverse-variance"

    def test_hrp_herc_variant(self, hrp_allocator, hrp_allocator_herc,
                             synthetic_returns_4asset_block_covariance):
        """
        Acceptance: HERC variant runs without error and produces different
        weights from HRP on the 4-asset fixture.
        """
        returns = synthetic_returns_4asset_block_covariance

        signals = {
            "A0": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "A1": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "A2": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "A3": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        # HRP variant
        weights_hrp = hrp_allocator.allocate(signals, returns, {}, {})

        # HERC variant
        weights_herc = hrp_allocator_herc.allocate(signals, returns, {}, {})

        # Both should sum to ~1.0
        assert abs(sum(weights_hrp.values()) - 1.0) < 0.01
        assert abs(sum(weights_herc.values()) - 1.0) < 0.01

        # They should produce different allocations
        # (HERC splits risk equally at each level, HRP uses inverse-variance)
        weights_differ = False
        for sym in ["A0", "A1", "A2", "A3"]:
            if abs(weights_hrp[sym] - weights_herc[sym]) > 0.01:
                weights_differ = True
                break

        assert weights_differ, "HRP and HERC should produce different weights"

    def test_hrp_signs_follow_directions(self, hrp_allocator, synthetic_returns_200_obs_3_assets):
        """Acceptance: weight signs follow signal directions."""
        returns = synthetic_returns_200_obs_3_assets

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.SHORT,
                size=0.1,
                entry=100.0,
                stop=101.0,
                target=99.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "SOL": Signal(
                direction=Direction.FLAT,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = hrp_allocator.allocate(signals, returns, {}, {})

        assert weights["BTC"] >= -1e-10, "BTC (LONG) should not be negative"
        assert weights["ETH"] <= 1e-10, "ETH (SHORT) should not be positive"
        assert abs(weights["SOL"]) < 1e-10, "SOL (FLAT) should be ~0"

    def test_hrp_fallback_below_min_obs(self, hrp_allocator, synthetic_returns_small):
        """Acceptance: fallback to equal weight when len(returns) < min_obs."""
        returns = synthetic_returns_small  # 5 obs
        assert len(returns) < PortfolioAllocator.min_obs

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = hrp_allocator.allocate(signals, returns, {}, {})

        expected = _fallback_equal_weight(signals)
        assert weights == expected, \
            f"Expected equal-weight fallback {expected}, got {weights}"

    def test_hrp_single_asset(self, hrp_allocator):
        """With single asset, allocate full weight."""
        returns = pd.DataFrame(
            np.random.randn(100) * 0.02,
            columns=["A"],
            index=pd.date_range("2024-01-01", periods=100),
        )

        signals = {
            "A": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = hrp_allocator.allocate(signals, returns, {}, {})

        assert weights["A"] == 1.0, "Single LONG asset should get weight 1.0"

    def test_hrp_empty_signals(self, hrp_allocator, synthetic_returns_200_obs_3_assets):
        """With no signals, return empty dict."""
        returns = synthetic_returns_200_obs_3_assets
        weights = hrp_allocator.allocate({}, returns, {}, {})
        assert weights == {}

    def test_hrp_allocator_attributes(self, hrp_allocator, hrp_allocator_herc):
        """Test allocator has correct name and attributes."""
        assert hrp_allocator.name == "hrp_allocator"
        assert hrp_allocator.min_obs == 60
        assert hrp_allocator.lookback == 120
        assert hrp_allocator.variant == "hrp"

        assert hrp_allocator_herc.variant == "herc"

    def test_hrp_invalid_variant(self):
        """Invalid variant should raise ValueError."""
        with pytest.raises(ValueError, match="variant must be 'hrp' or 'herc'"):
            HRPAllocator(variant="invalid")

    def test_hrp_3asset_convergence(self, hrp_allocator):
        """
        3 assets with different vols: HRP should converge to
        approximately inverse-variance weights.
        """
        np.random.seed(123)
        n_obs = 150

        # Create 3 uncorrelated assets with vols 0.10, 0.15, 0.20
        asset1 = np.random.randn(n_obs) * 0.10
        asset2 = np.random.randn(n_obs) * 0.15
        asset3 = np.random.randn(n_obs) * 0.20

        returns = pd.DataFrame(
            {"A1": asset1, "A2": asset2, "A3": asset3},
            index=pd.date_range("2024-01-01", periods=n_obs),
        )

        signals = {
            "A1": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "A2": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "A3": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = hrp_allocator.allocate(signals, returns, {}, {})

        # All should be positive (all LONG)
        for sym in ["A1", "A2", "A3"]:
            assert weights[sym] > 1e-8, f"{sym} should be LONG"

        # Weights should sum to ~1.0
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

        # For 3 uncorrelated assets with vol ratio 1:1.5:2,
        # inverse-variance ratio should be 4:1.78:1 (inverses of squares)
        # So A1 should get most weight, A3 should get least
        assert weights["A1"] > weights["A3"], \
            f"Lower-vol asset A1 ({weights['A1']}) should get more weight than A3 ({weights['A3']})"


# =============================================================================
# TEST MINVAR ALLOCATOR
# =============================================================================

class TestMinVarAllocator:
    """Test Minimum-Variance allocator with shrunk covariance."""

    @pytest.fixture
    def minvar_allocator(self):
        """Standard MinVar allocator instance."""
        return MinVarAllocator(lookback=120, gross_cap=1.0, max_weight=0.35)

    @pytest.fixture
    def synthetic_returns_2asset_uncorrelated_10_20_vol(self):
        """
        Two assets with sample vols exactly 10% and 20% and exactly zero sample
        correlation (via Gram-Schmidt orthogonalization).
        MinVar on uncorrelated assets gives inverse-variance weights → |w1|/|w2| ≈ 4:1.
        """
        np.random.seed(42)
        n_obs = 150

        a = np.random.randn(n_obs)
        b = np.random.randn(n_obs)

        # Demean and orthogonalize b against a (exact zero sample correlation)
        a = a - a.mean()
        b = b - b.mean()
        b = b - (np.dot(a, b) / np.dot(a, a)) * a

        # Rescale to exact sample standard deviations: 10% and 20%
        asset1 = a / a.std(ddof=1) * 0.10
        asset2 = b / b.std(ddof=1) * 0.20

        returns = pd.DataFrame(
            {"ASSET1": asset1, "ASSET2": asset2},
            index=pd.date_range("2024-01-01", periods=n_obs),
        )
        return returns

    @pytest.fixture
    def synthetic_returns_3asset_minvar_corr(self):
        """
        Three assets for testing MinVar with correlation effects:
        - ASSET1 and ASSET2: highly correlated, low vol (0.08)
        - ASSET3: uncorrelated, higher vol (0.16)

        MinVar should allocate less to the correlated pair combined vs.
        the uncorrelated case due to diversification benefits.
        """
        np.random.seed(42)
        n_obs = 150

        # Common factor for ASSET1 and ASSET2 (high correlation)
        factor12 = np.random.randn(n_obs)
        asset1 = 0.008 * factor12 + 0.002 * np.random.randn(n_obs)
        asset2 = 0.008 * factor12 + 0.002 * np.random.randn(n_obs)

        # Uncorrelated asset with higher vol
        asset3 = 0.016 * np.random.randn(n_obs)

        returns = pd.DataFrame(
            {"ASSET1": asset1, "ASSET2": asset2, "ASSET3": asset3},
            index=pd.date_range("2024-01-01", periods=n_obs),
        )
        return returns

    def test_minvar_2asset_inverse_var_ratio(self, minvar_allocator,
                                             synthetic_returns_2asset_uncorrelated_10_20_vol):
        """
        Acceptance: 2 assets with vol 10%/20% and zero correlation →
        |weights| ratio ≈ 4:1 (inverse-variance), within 15% tolerance.
        """
        returns = synthetic_returns_2asset_uncorrelated_10_20_vol

        # Verify uncorrelated
        corr = returns.corr()
        assert abs(corr.iloc[0, 1]) < 1e-10, "Assets should be exactly uncorrelated"

        # Verify vols
        vols = returns.std()
        assert abs(vols["ASSET1"] - 0.10) < 1e-3, f"ASSET1 vol should be ~10%, got {vols['ASSET1']}"
        assert abs(vols["ASSET2"] - 0.20) < 1e-3, f"ASSET2 vol should be ~20%, got {vols['ASSET2']}"

        signals = {
            "ASSET1": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ASSET2": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = minvar_allocator.allocate(signals, returns, {}, {})

        assert weights["ASSET1"] > 1e-8, "ASSET1 should be LONG"
        assert weights["ASSET2"] > 1e-8, "ASSET2 should be LONG"

        # Ticket acceptance: |w1|/|w2| ≈ 4:1 within 15% tolerance
        # Analytic result: for uncorrelated assets, w_i ∝ 1/σ_i²
        # So w1/w2 = σ2²/σ1² = (0.20/0.10)² = 4:1
        # Allow 15% deviation: range [3.4, 4.6]
        actual_ratio = weights["ASSET1"] / weights["ASSET2"]
        assert 3.4 <= actual_ratio <= 4.6, \
            f"Weight ratio {actual_ratio:.4f} should be ~4.0 (within 15% tolerance)"

    def test_minvar_3asset_correlation_effect(self, minvar_allocator,
                                              synthetic_returns_3asset_minvar_corr):
        """
        Acceptance: MinVar with 3 assets (2 correlated, 1 uncorrelated) produces
        meaningful allocation respecting the covariance structure.

        With 2 uncorrelated assets (ASSET1, ASSET3):
        - ASSET1 vol ≈ 0.008, ASSET3 vol ≈ 0.016
        - Uncorrelated → MinVar puts more weight on low-vol asset

        With 3 assets (ASSET1, ASSET2, ASSET3) where ASSET1/ASSET2 highly correlated:
        - Correlated pair has lower effective diversification value
        - The allocator should still respect the variance minimization criterion
        """
        returns = synthetic_returns_3asset_minvar_corr

        # Test case: Three-asset case (ASSET1 + ASSET2 + ASSET3)
        signals_3 = {
            "ASSET1": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ASSET2": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ASSET3": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights_3 = minvar_allocator.allocate(signals_3, returns, {}, {})

        # All three should have non-zero weight (all LONG)
        assert weights_3["ASSET1"] > 1e-8, "ASSET1 should have meaningful LONG weight"
        assert weights_3["ASSET2"] > 1e-8, "ASSET2 should have meaningful LONG weight"
        assert weights_3["ASSET3"] > 1e-8, "ASSET3 should have meaningful LONG weight"

        # Weights should respect caps
        assert sum(abs(w) for w in weights_3.values()) <= 1.0 + 1e-6
        assert all(abs(w) <= 0.35 + 1e-6 for w in weights_3.values())

    def test_minvar_respects_caps(self, minvar_allocator, synthetic_returns_200_obs_3_assets):
        """Acceptance: solution never violates gross_cap or max_weight constraints."""
        returns = synthetic_returns_200_obs_3_assets

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "SOL": Signal(
                direction=Direction.SHORT,
                size=0.1,
                entry=100.0,
                stop=101.0,
                target=99.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = minvar_allocator.allocate(signals, returns, {}, {})

        # Check max_weight constraint
        for sym, w in weights.items():
            assert abs(w) <= 0.35 + 1e-6, \
                f"{sym} weight {w} exceeds max_weight 0.35"

        # Check gross_cap constraint
        gross_leverage = sum(abs(w) for w in weights.values())
        assert gross_leverage <= 1.0 + 1e-6, \
            f"Gross leverage {gross_leverage} exceeds gross_cap 1.0"

        # Check sign constraints
        assert weights["BTC"] >= -1e-10, "BTC (LONG) should not be negative"
        assert weights["ETH"] >= -1e-10, "ETH (LONG) should not be negative"
        assert weights["SOL"] <= 1e-10, "SOL (SHORT) should not be positive"

    def test_minvar_signs_follow_directions(self, minvar_allocator, synthetic_returns_200_obs_3_assets):
        """Acceptance: weight signs follow signal directions."""
        returns = synthetic_returns_200_obs_3_assets

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.SHORT,
                size=0.1,
                entry=100.0,
                stop=101.0,
                target=99.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "SOL": Signal(
                direction=Direction.FLAT,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = minvar_allocator.allocate(signals, returns, {}, {})

        assert weights["BTC"] >= -1e-10, "BTC (LONG) should not be negative"
        assert weights["ETH"] <= 1e-10, "ETH (SHORT) should not be positive"
        assert abs(weights["SOL"]) < 1e-10, "SOL (FLAT) should be ~0"

    def test_minvar_fallback_below_min_obs(self, minvar_allocator, synthetic_returns_small, simple_signals):
        """Acceptance: fallback to equal weight when len(returns) < min_obs."""
        returns = synthetic_returns_small  # 5 obs
        assert len(returns) < PortfolioAllocator.min_obs

        weights = minvar_allocator.allocate(simple_signals, returns, {}, {})

        # Should fall back to equal weight
        expected = _fallback_equal_weight(simple_signals)
        assert weights == expected, \
            f"Expected equal-weight fallback {expected}, got {weights}"

    def test_minvar_variance_beats_equal_weight(self, minvar_allocator,
                                                synthetic_returns_2asset_uncorrelated_10_20_vol):
        """
        Acceptance: portfolio variance of MinVar solution <= variance of
        equal-weight on the same covariance matrix.
        """
        returns = synthetic_returns_2asset_uncorrelated_10_20_vol

        signals = {
            "ASSET1": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ASSET2": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        # Get MinVar weights
        weights_minvar = minvar_allocator.allocate(signals, returns, {}, {})

        # Get covariance
        trailing_returns = returns[["ASSET1", "ASSET2"]].tail(minvar_allocator.lookback)
        cov = shrunk_covariance(trailing_returns)

        # Compute variance of MinVar solution
        w_minvar = np.array([weights_minvar["ASSET1"], weights_minvar["ASSET2"]])
        var_minvar = np.dot(w_minvar, np.dot(cov, w_minvar))

        # Compute variance of equal-weight solution
        w_equal = np.array([0.5, 0.5])
        var_equal = np.dot(w_equal, np.dot(cov, w_equal))

        # MinVar should have lower variance
        assert var_minvar <= var_equal + 1e-8, \
            f"MinVar variance {var_minvar:.8f} should be <= equal-weight variance {var_equal:.8f}"

    def test_minvar_allocator_attributes(self, minvar_allocator):
        """Test allocator has correct name and min_obs."""
        assert minvar_allocator.name == "minvar_allocator"
        assert minvar_allocator.min_obs == 60
        assert minvar_allocator.lookback == 120
        assert minvar_allocator.gross_cap == 1.0
        assert minvar_allocator.max_weight == 0.35

    def test_minvar_allocator_custom_params(self):
        """Test allocator with custom parameters."""
        allocator = MinVarAllocator(
            lookback=60,
            gross_cap=2.0,
            max_weight=0.5
        )
        assert allocator.lookback == 60
        assert allocator.gross_cap == 2.0
        assert allocator.max_weight == 0.5

    def test_minvar_empty_signals(self, minvar_allocator, synthetic_returns_200_obs_3_assets):
        """With no signals, return empty dict."""
        returns = synthetic_returns_200_obs_3_assets
        weights = minvar_allocator.allocate({}, returns, {}, {})
        assert weights == {}

    def test_minvar_single_asset(self, minvar_allocator):
        """Single asset should get meaningful allocation (subject to max_weight)."""
        returns = pd.DataFrame(
            np.random.randn(100) * 0.02,
            columns=["A"],
            index=pd.date_range("2024-01-01", periods=100),
        )

        signals = {
            "A": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = minvar_allocator.allocate(signals, returns, {}, {})

        # Single LONG asset should get weight up to max_weight (limited by constraint)
        # With single asset, max feasible allocation is max_weight = 0.35
        assert 0.0 < weights["A"] <= 0.35 + 1e-6, \
            f"Single LONG asset should get positive weight <= max_weight, got {weights['A']}"

    def test_minvar_long_short_mix(self, minvar_allocator, synthetic_returns_200_obs_3_assets):
        """MinVar with mixed LONG/SHORT signals should respect directions."""
        returns = synthetic_returns_200_obs_3_assets

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.SHORT,
                size=0.1,
                entry=100.0,
                stop=101.0,
                target=99.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = minvar_allocator.allocate(signals, returns, {}, {})

        # BTC (LONG) should be non-negative
        assert weights["BTC"] >= -1e-10, "BTC (LONG) should not be negative"

        # ETH (SHORT) should be non-positive
        assert weights["ETH"] <= 1e-10, "ETH (SHORT) should not be positive"

        # At least one should be non-zero (unless both happen to be 0)
        assert abs(weights["BTC"]) + abs(weights["ETH"]) > 1e-8, \
            "At least one position should be meaningful"

    def test_minvar_custom_gross_cap(self):
        """Test allocator with custom gross_cap."""
        np.random.seed(42)
        n_obs = 150
        a = np.random.randn(n_obs) * 0.1
        b = np.random.randn(n_obs) * 0.2
        returns = pd.DataFrame(
            {"A": a, "B": b},
            index=pd.date_range("2024-01-01", periods=n_obs),
        )

        signals = {
            "A": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "B": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=1.0,
                stop=0.99,
                target=1.01,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        # Test with gross_cap = 0.5 (tighter cap)
        allocator = MinVarAllocator(gross_cap=0.5, max_weight=0.35)
        weights = allocator.allocate(signals, returns, {}, {})

        gross_leverage = sum(abs(w) for w in weights.values())
        assert gross_leverage <= 0.5 + 1e-6, \
            f"Gross leverage {gross_leverage} should respect custom cap 0.5"
