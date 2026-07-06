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
)
from kairos_backtest import Signal, Direction


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
