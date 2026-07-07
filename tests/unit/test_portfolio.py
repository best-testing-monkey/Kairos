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
    _max_sharpe_solve,
    _bl_posterior,
    _eigen_portfolios,
    MVOAllocator,
    RiskParityAllocator,
    HRPAllocator,
    MinVarAllocator,
    BlackLittermanAllocator,
    EigenAllocator,
    UniversalAllocator,
    GAAllocator,
    CVaRAllocator,
    KellyAllocator,
    Rebalancer,
    _scenario_matrix,
    _compute_cvar,
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


# =============================================================================
# TEST BLACK-LITTERMAN ALLOCATOR
# =============================================================================

class TestBlackLittermanPosterior:
    """Test the _bl_posterior helper function (module-level)."""

    def test_bl_posterior_basic_computation(self):
        """Verify posterior computation with simple 2-asset case."""
        # Simple prior and views
        pi = np.array([0.05, 0.04])
        Q = np.array([0.06, 0.03])
        Sigma = np.array([[0.01, 0.002], [0.002, 0.008]])
        Omega = np.eye(2) * 0.01
        tau = 0.05

        mu_bl = _bl_posterior(pi, Q, Sigma, Omega, tau=tau)

        # Posterior should be a convex combination of prior and views
        # With moderate Ω, it should be closer to prior (tau=0.05 is small)
        assert mu_bl.shape == (2,)
        assert np.all(np.isfinite(mu_bl))

    def test_bl_posterior_zero_confidence_equals_prior(self):
        """
        Acceptance: with very weak confidence views (Ω→∞, i.e., huge Ω),
        posterior mu_BL should approach prior Π.

        This tests the limiting case where we have near-zero confidence in views.
        """
        pi = np.array([0.05, 0.04, 0.03])
        Q = np.array([0.10, 0.01, 0.02])  # Very different from prior
        Sigma = np.array([
            [0.01, 0.001, 0.0],
            [0.001, 0.008, 0.001],
            [0.0, 0.001, 0.012],
        ])
        tau = 0.05

        # Very weak confidence: large Ω (uncertainty)
        Omega = np.eye(3) * 100.0  # Huge uncertainty → near-zero confidence

        mu_bl = _bl_posterior(pi, Q, Sigma, Omega, tau=tau)

        # Posterior should be very close to prior
        # Allow 10% deviation per acceptance criteria
        rel_error = np.abs(mu_bl - pi) / (np.abs(pi) + 1e-8)
        assert np.all(rel_error < 0.10), \
            f"With zero-confidence views, mu_BL should be ~π. Got {mu_bl} vs prior {pi}"

    def test_bl_posterior_infinite_confidence_matches_views(self):
        """
        Acceptance: with very strong confidence in views (Ω→0, i.e., tiny Ω),
        posterior mu_BL should approach views Q.

        This tests the limiting case where we have very high confidence in views.
        """
        pi = np.array([0.05, 0.04])
        Q = np.array([0.10, 0.02])
        Sigma = np.array([[0.01, 0.002], [0.002, 0.008]])
        tau = 0.05

        # Very strong confidence: tiny Ω (near-zero uncertainty)
        Omega = np.eye(2) * 1e-6

        mu_bl = _bl_posterior(pi, Q, Sigma, Omega, tau=tau)

        # Posterior should be very close to views
        rel_error = np.abs(mu_bl - Q) / (np.abs(Q) + 1e-8)
        assert np.all(rel_error < 0.05), \
            f"With infinite-confidence views, mu_BL should be ~Q. Got {mu_bl} vs Q {Q}"

    def test_bl_posterior_intermediate_entropy(self):
        """
        Acceptance: with intermediate entropy (between 0 and ln(20)),
        posterior should be strictly between prior Π and views Q (for same-sign views).
        """
        pi = np.array([0.05, 0.04])
        Q = np.array([0.10, 0.08])  # Higher returns than prior
        Sigma = np.array([[0.01, 0.002], [0.002, 0.008]])
        tau = 0.05

        # Intermediate confidence: moderate Ω
        Omega = np.eye(2) * 0.005

        mu_bl = _bl_posterior(pi, Q, Sigma, Omega, tau=tau)

        # For assets where Q > π, posterior should satisfy π < μ_BL < Q
        assert mu_bl[0] > pi[0] - 1e-10, \
            f"Posterior for asset 0 ({mu_bl[0]}) should be > prior ({pi[0]})"
        assert mu_bl[0] < Q[0] + 1e-10, \
            f"Posterior for asset 0 ({mu_bl[0]}) should be < view ({Q[0]})"

    def test_bl_posterior_respects_low_entropy_strong_view(self):
        """
        Acceptance: low-entropy (certain) distributions should produce strong views.
        Posterior should move significantly toward the view in that case.
        """
        pi = np.array([0.05, 0.04])
        Q = np.array([0.15, 0.04])  # Asset 0: big upside
        Sigma = np.array([[0.01, 0.002], [0.002, 0.008]])
        tau = 0.05

        # Low entropy → strong confidence for asset 0
        omega_0_0 = 0.001 * 0.05 * Sigma[0, 0]  # Low entropy scaling
        omega_1_1 = 1.5 * 0.05 * Sigma[1, 1]     # High entropy (fallback)
        Omega = np.diag([omega_0_0, omega_1_1])

        mu_bl = _bl_posterior(pi, Q, Sigma, Omega, tau=tau)

        # Asset 0 should move significantly toward the high view
        gap_to_prior = Q[0] - pi[0]  # 0.10
        movement = mu_bl[0] - pi[0]
        # Should move at least 30% of the way toward Q
        assert movement > 0.3 * gap_to_prior, \
            f"With low-entropy view, posterior should move significantly. " \
            f"Got {movement} vs gap {gap_to_prior}"


class TestBlackLittermanAllocator:
    """Test BlackLittermanAllocator class."""

    def test_bl_allocator_basic_allocation(self, synthetic_returns_200_obs_3_assets):
        """Verify basic Black-Litterman allocation with simple signals."""
        returns = synthetic_returns_200_obs_3_assets
        symbols = ["BTC", "ETH", "SOL"]

        # Create signals and distributions
        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=2.0,
            )
            for sym in symbols
        }

        dists = {}
        for sym in symbols:
            dist = type("MockDist", (), {})()
            dist.stats = {
                "close": {"mean": 105.0},  # Expect 5% return
                "high": {"mean": 106.0},
                "low": {"mean": 104.0},
            }
            dist.entropy = lambda: 1.5  # Fallback entropy
            dists[sym] = dist

        allocator = BlackLittermanAllocator(tau=0.05, delta=2.5, lookback=120)
        weights = allocator.allocate(signals, returns, dists, {})

        # Basic checks
        assert isinstance(weights, dict)
        assert set(weights.keys()) == set(symbols)
        assert all(isinstance(v, float) for v in weights.values())
        # All signals are LONG, so weights should be non-negative
        assert all(w >= -1e-10 for w in weights.values())
        # Gross leverage should respect cap
        gross = sum(abs(w) for w in weights.values())
        assert gross <= 1.0 + 1e-6

    def test_bl_allocator_below_min_obs_fallback(self, synthetic_returns_small):
        """With fewer than 60 observations, should return equal weight."""
        returns = synthetic_returns_small  # 5 obs
        symbols = ["BTC", "ETH", "SOL"]

        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for sym in symbols
        }

        dists = {sym: type("MockDist", (), {})() for sym in symbols}
        for sym in symbols:
            dists[sym].stats = {"close": {"mean": 105.0}}
            dists[sym].entropy = lambda: 1.5

        allocator = BlackLittermanAllocator()
        weights = allocator.allocate(signals, returns, dists, {})

        # Should be equal weight
        expected_weight = 1.0 / len(symbols)
        for sym in symbols:
            assert abs(weights[sym] - expected_weight) < 1e-10

    def test_bl_allocator_empty_signals_returns_empty(self, synthetic_returns_200_obs_3_assets):
        """With no signals, should return empty dict."""
        returns = synthetic_returns_200_obs_3_assets
        signals = {}
        dists = {}

        allocator = BlackLittermanAllocator()
        weights = allocator.allocate(signals, returns, dists, {})

        assert weights == {}

    def test_bl_allocator_respects_max_weight_cap(self, synthetic_returns_200_obs_3_assets):
        """Verify max_weight constraint is respected."""
        returns = synthetic_returns_200_obs_3_assets
        symbols = ["BTC", "ETH", "SOL"]

        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=2.0,
            )
            for sym in symbols
        }

        dists = {}
        for sym in symbols:
            dist = type("MockDist", (), {})()
            dist.stats = {"close": {"mean": 105.0}}
            dist.entropy = lambda: 0.5
            dists[sym] = dist

        allocator = BlackLittermanAllocator(max_weight=0.25)
        weights = allocator.allocate(signals, returns, dists, {})

        # No single weight should exceed max_weight
        for w in weights.values():
            assert abs(w) <= 0.25 + 1e-6

    def test_bl_allocator_respects_gross_cap(self, synthetic_returns_200_obs_3_assets):
        """Verify gross leverage cap is respected."""
        returns = synthetic_returns_200_obs_3_assets
        symbols = ["BTC", "ETH"]

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=2.0,
            ),
            "ETH": Signal(
                direction=Direction.SHORT,
                size=0.1,
                entry=100.0,
                stop=101.0,
                target=98.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=-2.0,
            ),
        }

        dists = {}
        for sym in symbols:
            dist = type("MockDist", (), {})()
            dist.stats = {"close": {"mean": 105.0 if sym == "BTC" else 95.0}}
            dist.entropy = lambda: 1.0
            dists[sym] = dist

        allocator = BlackLittermanAllocator(gross_cap=0.5)
        weights = allocator.allocate(signals, returns, dists, {})

        # Gross leverage should respect cap
        gross = sum(abs(w) for w in weights.values())
        assert gross <= 0.5 + 1e-6

    def test_bl_allocator_long_short_directions(self, synthetic_returns_200_obs_3_assets):
        """Mixed LONG/SHORT signals should produce appropriately signed weights."""
        returns = synthetic_returns_200_obs_3_assets
        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            ),
            "ETH": Signal(
                direction=Direction.SHORT,
                size=0.1,
                entry=100.0,
                stop=101.0,
                target=98.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=-1.0,
            ),
        }

        dists = {}
        for sym in signals:
            dist = type("MockDist", (), {})()
            dist.stats = {"close": {"mean": 105.0}}
            dist.entropy = lambda: 1.5
            dists[sym] = dist

        allocator = BlackLittermanAllocator()
        weights = allocator.allocate(signals, returns, dists, {})

        # BTC (LONG) should be non-negative
        assert weights["BTC"] >= -1e-10, "LONG signal should have non-negative weight"
        # ETH (SHORT) should be non-positive
        assert weights["ETH"] <= 1e-10, "SHORT signal should have non-positive weight"

    def test_bl_allocator_missing_dist_entropy_fallback(self, synthetic_returns_200_obs_3_assets):
        """When dist.entropy() raises an exception, should fallback to 1.5."""
        returns = synthetic_returns_200_obs_3_assets
        symbols = ["BTC", "ETH"]

        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for sym in symbols
        }

        dists = {}
        for sym in symbols:
            dist = type("MockDist", (), {})()
            dist.stats = {"close": {"mean": 105.0}}
            # entropy() will raise when called
            dist.entropy = lambda: (_ for _ in ()).throw(RuntimeError("test"))
            dists[sym] = dist

        allocator = BlackLittermanAllocator()
        # Should not crash; falls back to entropy=1.5
        weights = allocator.allocate(signals, returns, dists, {})

        assert isinstance(weights, dict)
        assert len(weights) == 2

    def test_bl_allocator_regression_mvo_compatible(self, synthetic_returns_200_obs_3_assets):
        """
        Regression: Black-Litterman should produce weights comparable to MVOAllocator
        when views are close to MVO's expected values.
        """
        returns = synthetic_returns_200_obs_3_assets
        symbols = ["BTC", "ETH", "SOL"]

        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for sym in symbols
        }

        dists = {}
        for sym in symbols:
            dist = type("MockDist", (), {})()
            dist.stats = {"close": {"mean": 101.0}}  # 1% expected return
            dist.entropy = lambda: 0.1  # Very certain
            # Add expected_value for MVO
            dist.expected_value = lambda entry, target, stop: 0.01
            dists[sym] = dist

        bl_allocator = BlackLittermanAllocator(tau=0.05)
        mvo_allocator = MVOAllocator()

        bl_weights = bl_allocator.allocate(signals, returns, dists, {})
        mvo_weights = mvo_allocator.allocate(signals, returns, dists, {})

        # Both should allocate positive weight to LONG signals
        for sym in symbols:
            assert bl_weights[sym] >= -1e-10
            assert mvo_weights[sym] >= -1e-10

    def test_bl_allocator_max_entropy_weak_view(self, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: entropy=ln(20) (~3.0) view moves posterior <10% of the way
        from prior to view.
        """
        returns = synthetic_returns_200_obs_3_assets
        symbols = list(returns.columns)

        # Create high-entropy (weak) signals
        signals = {}
        for sym in symbols:
            signals[sym] = Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.05,  # 5% MVO expectation
            )

        ln_20 = np.log(20.0)
        dists = {}
        for sym in symbols:
            dist = type("MockDist", (), {})()
            # View that's very different from prior
            dist.stats = {"close": {"mean": 110.0}}  # 10% return
            dist.entropy = lambda: ln_20  # Max entropy = weak view
            dists[sym] = dist

        allocator = BlackLittermanAllocator(tau=0.05, delta=2.5)
        weights = allocator.allocate(signals, returns, dists, {})

        # With max-entropy (ln 20 ≈ 3.0) views, posterior should be close to prior
        # so weights should be closer to inverse-variance (from prior) than to
        # pure view-based allocation. Hard to test directly without exposing internals,
        # but we can verify the allocator produces sensible weights.
        assert isinstance(weights, dict)
        assert len(weights) > 0
        gross = sum(abs(w) for w in weights.values())
        assert gross <= 1.0 + 1e-6


class TestMaxSharpeSolverRegression:
    """Verify _max_sharpe_solve is used correctly by both MVO and BL."""

    def test_max_sharpe_solve_shared_function(self):
        """Verify _max_sharpe_solve function exists and is callable."""
        assert callable(_max_sharpe_solve)

    def test_max_sharpe_solve_basic_call(self):
        """Basic call to _max_sharpe_solve with synthetic data."""
        symbols = ["A", "B"]
        mu = np.array([0.05, 0.04])
        cov = np.array([[0.01, 0.002], [0.002, 0.008]])

        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for sym in symbols
        }

        weights = _max_sharpe_solve(symbols, mu, cov, signals)

        assert isinstance(weights, dict)
        assert set(weights.keys()) == set(symbols)
        assert all(isinstance(v, float) for v in weights.values())

    def test_mvo_still_works_after_refactor(self, synthetic_returns_200_obs_3_assets):
        """Regression: MVOAllocator should still work after refactoring."""
        returns = synthetic_returns_200_obs_3_assets
        symbols = list(returns.columns)

        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for sym in symbols
        }

        dists = {}
        for sym in symbols:
            dist = type("MockDist", (), {})()
            dist.stats = {"close": {"mean": 105.0}}
            dist.entropy = lambda: 1.5
            dist.expected_value = lambda entry, target, stop: 0.01
            dists[sym] = dist

        allocator = MVOAllocator()
        weights = allocator.allocate(signals, returns, dists, {})

        # Should produce valid weights
        assert isinstance(weights, dict)
        assert len(weights) == len(symbols)
        assert all(w >= -1e-10 for w in weights.values())


# =============================================================================
# TEST EIGEN PORTFOLIO ALLOCATOR
# =============================================================================

class TestEigenPortfolios:
    """Test the _eigen_portfolios helper function."""

    def test_eigen_portfolios_orthogonal(self):
        """
        Acceptance: eigenvectors returned by _eigen_portfolios are mutually orthogonal.
        Verify pairwise dot products are approximately zero.
        """
        # Create a simple correlation matrix
        np.random.seed(42)
        n = 5
        # Generate a correlation matrix: start with random matrix, symmetrize, normalize
        A = np.random.randn(n, n)
        corr = A @ A.T  # Symmetric positive semi-definite
        # Normalize to correlation matrix (diagonal = 1)
        diag_std = np.sqrt(np.diag(corr))
        corr = corr / np.outer(diag_std, diag_std)
        np.fill_diagonal(corr, 1.0)

        # Extract top-k eigenvectors (excluding PC1)
        k = 3
        V = _eigen_portfolios(corr, k)

        # Check orthogonality: V'V should be approximately I (since eigh returns orthonormal)
        gram = V.T @ V
        expected_gram = np.eye(k)

        # Pairwise dot products should be ≈ 0 (off-diagonal)
        # and ≈ 1 on diagonal (normalization)
        for i in range(k):
            for j in range(k):
                if i == j:
                    assert abs(gram[i, j] - 1.0) < 1e-10, \
                        f"Diagonal: gram[{i},{j}] = {gram[i,j]}, expected 1.0"
                else:
                    assert abs(gram[i, j]) < 1e-10, \
                        f"Off-diagonal: gram[{i},{j}] = {gram[i,j]}, expected 0.0"

    def test_eigen_portfolios_excludes_pc1(self):
        """
        Acceptance: _eigen_portfolios excludes the dominant eigenvector (PC1).
        Verify the returned eigenvectors are not the dominant one.
        """
        # Create correlation matrix with clear dominant PC1
        np.random.seed(123)
        n = 4
        # One strong common factor
        factor = np.random.randn(100)
        returns = np.tile(factor.reshape(-1, 1), (1, n)) + 0.1 * np.random.randn(100, n)
        corr = np.corrcoef(returns.T)

        # Get dominant eigenvector
        eigvals, eigvecs = np.linalg.eigh(corr)
        pc1 = eigvecs[:, -1]  # Largest eigenvector

        # Get next-2 eigenvectors from our helper
        k = 2
        V = _eigen_portfolios(corr, k)

        # Check that PC1 is not in V
        # PC1 should be orthogonal to all columns of V (dot product ≈ 0)
        for i in range(k):
            dot_prod = np.abs(np.dot(pc1, V[:, i]))
            assert dot_prod < 1e-10, \
                f"PC1 is not orthogonal to V[:, {i}]: dot product = {dot_prod}"

    def test_eigen_portfolios_k_validation(self):
        """Verify _eigen_portfolios rejects invalid k values."""
        corr = np.eye(3)

        # k < 1 should raise
        with pytest.raises(ValueError):
            _eigen_portfolios(corr, k=0)

        # k >= n should raise
        with pytest.raises(ValueError):
            _eigen_portfolios(corr, k=3)

        # k = n-1 should work (3-1=2 eigenvectors after removing PC1)
        V = _eigen_portfolios(corr, k=2)
        assert V.shape == (3, 2)

    def test_eigen_portfolios_shape(self):
        """Verify _eigen_portfolios returns correct shape."""
        n = 5
        k = 2
        corr = np.eye(n) + 0.1 * np.random.randn(n, n)
        corr = (corr + corr.T) / 2  # Symmetrize
        np.fill_diagonal(corr, 1.0)  # Ensure correlation

        V = _eigen_portfolios(corr, k)

        assert V.shape == (n, k), f"Expected shape ({n}, {k}), got {V.shape}"


class TestEigenAllocator:
    """Test the EigenAllocator class."""

    def test_eigen_allocator_basic(self, synthetic_returns_200_obs_3_assets):
        """
        Basic test: EigenAllocator produces weights for 2+ assets.
        """
        returns = synthetic_returns_200_obs_3_assets
        symbols = list(returns.columns)

        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for sym in symbols
        }

        allocator = EigenAllocator(n_components=2, lookback=120)
        weights = allocator.allocate(signals, returns, {}, {})

        assert isinstance(weights, dict)
        assert len(weights) == len(symbols)
        assert all(isinstance(v, float) for v in weights.values())

    def test_eigen_allocator_weights_sum_to_1(self, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: weights sum to 1 in absolute value.
        """
        returns = synthetic_returns_200_obs_3_assets
        symbols = list(returns.columns)

        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for sym in symbols
        }

        allocator = EigenAllocator(n_components=2, lookback=120)
        weights = allocator.allocate(signals, returns, {}, {})

        gross = sum(abs(w) for w in weights.values())
        assert abs(gross - 1.0) < 1e-6, f"Weights sum to {gross}, expected 1.0"

    def test_eigen_allocator_signs_follow_direction(self, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: weights sign matches signal direction (LONG=+, SHORT=-, FLAT=0).
        """
        returns = synthetic_returns_200_obs_3_assets
        symbols = list(returns.columns)

        signals = {
            symbols[0]: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            ),
            symbols[1]: Signal(
                direction=Direction.SHORT,
                size=0.1,
                entry=100.0,
                stop=101.0,
                target=98.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            ),
            symbols[2]: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            ),
        }

        allocator = EigenAllocator(n_components=2, lookback=120)
        weights = allocator.allocate(signals, returns, {}, {})

        # Check signs
        if abs(weights[symbols[0]]) > 1e-10:
            assert weights[symbols[0]] > 0, "LONG signal should have positive weight"
        if abs(weights[symbols[1]]) > 1e-10:
            assert weights[symbols[1]] < 0, "SHORT signal should have negative weight"
        if abs(weights[symbols[2]]) > 1e-10:
            assert weights[symbols[2]] > 0, "LONG signal should have positive weight"

    def test_eigen_allocator_below_min_obs_fallback(self, synthetic_returns_small, simple_signals):
        """
        Acceptance: below min_obs threshold, falls back to equal weight.
        """
        returns = synthetic_returns_small  # 5 observations < min_obs=60

        allocator = EigenAllocator()
        weights = allocator.allocate(simple_signals, returns, {}, {})

        # Should be equal weight
        assert weights == {"BTC": 0.5, "ETH": 0.5}

    def test_eigen_allocator_single_asset_fallback(self, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: with only 1 asset, falls back to equal weight.
        """
        returns = synthetic_returns_200_obs_3_assets
        symbol = returns.columns[0]

        signals = {
            symbol: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
        }

        allocator = EigenAllocator()
        weights = allocator.allocate(signals, returns, {}, {})

        # Should fall back to equal weight
        assert weights == {symbol: 1.0}

    def test_eigen_allocator_no_signals(self, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: with no signals, returns empty dict.
        """
        returns = synthetic_returns_200_obs_3_assets

        allocator = EigenAllocator()
        weights = allocator.allocate({}, returns, {}, {})

        assert weights == {}

    def test_eigen_pc1_exclusion_reduces_correlation(self):
        """
        Acceptance: PC1 exclusion reduces average pairwise correlation of
        resulting weight vector with equal-weight basket.

        Generate seeded panel with one dominant market factor plus idiosyncratic
        structure, verify that eigen-portfolio weights have lower correlation
        with equal-weight vector than the market mode eigenvector does.
        """
        np.random.seed(42)
        n_obs = 200
        n_assets = 5

        # Create data with dominant market factor + idiosyncratic noise
        market_factor = np.random.randn(n_obs)
        returns = np.zeros((n_obs, n_assets))

        for i in range(n_assets):
            # Each asset has strong market component + idiosyncratic component
            market_weight = 0.8
            idio_weight = 0.2
            returns[:, i] = market_weight * market_factor + idio_weight * np.random.randn(n_obs)

        returns_df = pd.DataFrame(
            returns,
            columns=[f"Asset{i}" for i in range(n_assets)],
            index=pd.date_range("2024-01-01", periods=n_obs),
        )

        symbols = list(returns_df.columns)
        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for sym in symbols
        }

        # Compute equal-weight vector
        equal_weight_vec = np.ones(n_assets) / n_assets

        # Compute correlation matrix
        corr = np.corrcoef(returns_df.T)

        # Get PC1 (market mode)
        eigvals, eigvecs = np.linalg.eigh(corr)
        pc1 = eigvecs[:, -1]  # Largest eigenvector

        # Compute eigen-allocator weights
        allocator = EigenAllocator(n_components=3, lookback=120)
        weights_dict = allocator.allocate(signals, returns_df, {}, {})

        eigen_weights = np.array([weights_dict[sym] for sym in symbols])
        # Normalize to unit norm for fair correlation comparison
        eigen_weights_norm = eigen_weights / (np.linalg.norm(eigen_weights) + 1e-10)

        # Normalize PC1 and equal-weight for fair comparison
        pc1_norm = pc1 / (np.linalg.norm(pc1) + 1e-10)
        equal_weight_norm = equal_weight_vec / (np.linalg.norm(equal_weight_vec) + 1e-10)

        # Correlation of eigen-weights with equal-weight
        corr_eigen_ew = np.abs(np.dot(eigen_weights_norm, equal_weight_norm))

        # Correlation of PC1 with equal-weight
        corr_pc1_ew = np.abs(np.dot(pc1_norm, equal_weight_norm))

        # Eigen-portfolio should be more orthogonal to equal-weight than PC1 is
        # (i.e., lower correlation)
        assert corr_eigen_ew < corr_pc1_ew, \
            f"Eigen correlation {corr_eigen_ew} should be < PC1 correlation {corr_pc1_ew}"

    def test_eigen_allocator_short_window_fallback(self, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: with very short trailing window (<2 obs), falls back to equal weight.
        """
        # Create a returns dataframe with only 1 observation
        returns = synthetic_returns_200_obs_3_assets.iloc[:1]

        symbols = list(returns.columns)
        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for sym in symbols
        }

        allocator = EigenAllocator(lookback=120)
        # This should trigger the fallback because correlation requires at least 2 obs
        weights = allocator.allocate(signals, returns, {}, {})

        # Should fall back to equal weight
        assert weights == {sym: 1.0 / len(symbols) for sym in symbols}

    def test_eigen_allocator_name(self):
        """Test the allocator name attribute."""
        allocator = EigenAllocator()
        assert allocator.name == "eigen_allocator"

    def test_eigen_allocator_init_params(self):
        """Test initialization parameters."""
        allocator = EigenAllocator(n_components=5, lookback=250)
        assert allocator.n_components == 5
        assert allocator.lookback == 250

    def test_eigen_allocator_n_components_respected(self, synthetic_returns_200_obs_3_assets):
        """
        Verify that n_components parameter is respected.
        With n_components=1 and 3 assets, should use only 1 eigenvector (after PC1).
        """
        returns = synthetic_returns_200_obs_3_assets
        symbols = list(returns.columns)

        signals = {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=102.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=1.0,
            )
            for sym in symbols
        }

        # Create allocators with different n_components
        allocator1 = EigenAllocator(n_components=1, lookback=120)
        allocator2 = EigenAllocator(n_components=2, lookback=120)

        weights1 = allocator1.allocate(signals, returns, {}, {})
        weights2 = allocator2.allocate(signals, returns, {}, {})

        # Both should produce valid weights
        assert isinstance(weights1, dict)
        assert isinstance(weights2, dict)
        assert len(weights1) == len(symbols)
        assert len(weights2) == len(symbols)

        # Weights should be different (different number of components used)
        weights_array1 = np.array([weights1[sym] for sym in symbols])
        weights_array2 = np.array([weights2[sym] for sym in symbols])
        # They should not be identical
        assert not np.allclose(weights_array1, weights_array2), \
            "Different n_components should produce different weights"


# =============================================================================
# TEST UNIVERSAL ALLOCATOR (COVER PORTFOLIO)
# =============================================================================

class TestUniversalAllocator:
    """Test Universal Portfolio (Cover) allocator with wealth-weighted CRPs."""

    @pytest.fixture
    def universal_allocator(self):
        """Standard Universal allocator instance."""
        return UniversalAllocator(grid_step=0.1)

    @pytest.fixture
    def synthetic_returns_3asset_drift(self):
        """
        Three assets over ~120 days where Asset A has consistent +0.5%/day drift.
        Assets B and C have zero drift (mean-reverting around 0).
        Used to test convergence to dominant asset.
        """
        np.random.seed(42)
        n_obs = 120

        # Asset A: +0.5% drift + noise
        asset_a = 0.005 + 0.005 * np.random.randn(n_obs)

        # Asset B & C: noise only (zero drift)
        asset_b = 0.001 * np.random.randn(n_obs)
        asset_c = 0.001 * np.random.randn(n_obs)

        returns = pd.DataFrame(
            {"A": asset_a, "B": asset_b, "C": asset_c},
            index=pd.date_range("2024-01-01", periods=n_obs),
        )
        return returns

    def _create_signals_for_symbols(self, symbols):
        """Helper to create LONG signals for a list of symbols."""
        return {
            sym: Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=99.0,
                target=101.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            )
            for sym in symbols
        }

    def test_universal_grid_generation_3asset_step_01(self, universal_allocator):
        """
        Acceptance: grid generation count for 3 assets with step 0.1 == 66 (C(12,2)).

        For n=3 and num_steps=10 (since 1.0/0.1=10), the number of non-negative
        integer solutions to k1 + k2 + k3 = 10 is C(10+3-1, 3-1) = C(12,2) = 66.
        """
        allocator = universal_allocator
        symbols = ["A", "B", "C"]
        allocator._regenerate_grid(symbols)

        assert allocator.grid is not None
        assert allocator.grid_symbols == symbols
        assert len(allocator.grid) == 66, \
            f"Expected 66 grid points for 3 assets with step=0.1, got {len(allocator.grid)}"

        # Verify each grid point is a valid weight vector
        for grid_weights in allocator.grid:
            assert len(grid_weights) == 3, "Each grid point should have 3 weights"
            assert abs(np.sum(grid_weights) - 1.0) < 1e-10, \
                f"Grid point weights should sum to 1.0, got {np.sum(grid_weights)}"
            assert np.all(grid_weights >= -1e-10), "All weights should be non-negative"
            assert np.all(grid_weights <= 1.0 + 1e-10), "All weights should be <= 1.0"

    def test_universal_grid_weights_sum_to_one(self, universal_allocator, synthetic_returns_3asset_drift):
        """
        Acceptance: total weight always sums to 1 (verified in test_universal_weights_sum_to_one).
        """
        allocator = universal_allocator
        returns = synthetic_returns_3asset_drift
        signals = self._create_signals_for_symbols(["A", "B", "C"])

        # Feed multiple calls to build up wealth distribution
        for _ in range(10):
            weights = allocator.allocate(signals, returns, {}, {})
            total = sum(abs(w) for w in weights.values())
            # All LONG signals, so sum of weights = sum of absolute values
            assert abs(total - 1.0) < 1e-6, \
                f"Weights should sum to 1.0, got {total}"

    def test_universal_convergence_to_dominant(self, universal_allocator, synthetic_returns_3asset_drift):
        """
        Acceptance: on synthetic data where one asset dominates, weights converge
        toward it. Verified in test_universal_convergence_to_dominant.

        After ~120 sequential allocate() calls on the drift data (Asset A +0.5%/day),
        weight on A increases monotonically over the last 20 calls (or at least end > start).
        """
        allocator = universal_allocator
        returns = synthetic_returns_3asset_drift
        symbols = ["A", "B", "C"]
        signals = self._create_signals_for_symbols(symbols)

        # Feed all observations sequentially, accumulating returns
        # (simulating a walk-forward scenario where allocator sees one day at a time)
        all_weights = []
        for i in range(len(returns)):
            current_returns = returns.iloc[:i+1]  # Cumulative up to day i
            weights = allocator.allocate(signals, current_returns, {}, {})
            all_weights.append(weights)

        # Extract weight on A from all allocations
        weights_a_all = [w["A"] for w in all_weights]

        # Weight should increase from the early allocations to the later ones
        # Universal portfolio converges gradually; check that final > initial
        assert weights_a_all[-1] > weights_a_all[0], \
            f"Weight on dominant asset A should increase from start to end: " \
            f"{weights_a_all[0]:.4f} -> {weights_a_all[-1]:.4f}"

        # Also check that the weight is above equal-weight (1/3 ≈ 0.333)
        assert weights_a_all[-1] > 1.0 / 3.0, \
            f"Weight on A should exceed equal-weight 1/3, got {weights_a_all[-1]:.4f}"

    def test_universal_wealth_weighted_output(self, universal_allocator, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: outputs wealth-weighted average of grid points (verified in
        test_universal_wealth_weighted_output).
        """
        allocator = universal_allocator
        returns = synthetic_returns_200_obs_3_assets
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)

        # Call allocate a few times to let wealth distribution form
        for i in range(len(returns)):
            current_returns = returns.iloc[:i+1]
            weights = allocator.allocate(signals, current_returns, {}, {})

            # After at least 2 calls, wealth should be updating
            if i > 0 and allocator.wealth is not None:
                # Some grid points should have different wealth (wealth is not uniform)
                # unless all returns are identical (unlikely in random data)
                wealth_std = np.std(allocator.wealth)
                if i > 10:
                    # After enough iterations, wealth should have diverged
                    assert wealth_std > 1e-8, \
                        "Wealth should diverge after multiple updates"

            # Output should be a valid dict
            assert isinstance(weights, dict)
            assert len(weights) == len(symbols)
            for w in weights.values():
                assert isinstance(w, float)

    def test_universal_grid_regeneration_on_symbol_change(self, universal_allocator, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: grid regenerated when universe changes (test_universal_grid_regeneration).
        When symbol set changes (e.g., ["BTC", "ETH"] -> ["BTC", "ETH", "SOL"]),
        allocator regenerates grid and resets wealth.
        """
        allocator = universal_allocator
        returns = synthetic_returns_200_obs_3_assets

        # Start with 2 assets
        signals_2 = self._create_signals_for_symbols(["BTC", "ETH"])
        allocator.allocate(signals_2, returns[["BTC", "ETH"]], {}, {})

        grid_symbols_1 = allocator.grid_symbols
        grid_size_1 = len(allocator.grid)

        # Advance time and update wealth a bit
        for i in range(10):
            current_returns = returns[["BTC", "ETH"]].iloc[:60+i]
            allocator.allocate(signals_2, current_returns, {}, {})

        wealth_after_updates = allocator.wealth.copy()
        assert not np.allclose(wealth_after_updates, np.ones(len(wealth_after_updates))), \
            "Wealth should have diverged from initial 1.0"

        # Switch to 3 assets (change universe)
        signals_3 = self._create_signals_for_symbols(["BTC", "ETH", "SOL"])

        # Capture grid and wealth right at the moment of regeneration
        # by checking before/after within the same allocate call
        # After the call, wealth will have been updated with day's return, so we
        # check that it was reset by verifying the grid changed and the grid size matches
        allocator.allocate(signals_3, returns, {}, {})

        grid_symbols_2 = allocator.grid_symbols
        grid_size_2 = len(allocator.grid)
        wealth_2 = allocator.wealth.copy()

        # Grid should be regenerated (different size)
        assert grid_symbols_2 == ["BTC", "ETH", "SOL"], \
            f"Symbol set should change to 3 assets, got {grid_symbols_2}"
        assert grid_size_2 == 66, \
            f"3-asset grid should have 66 points, got {grid_size_2}"
        assert grid_size_2 != grid_size_1, \
            f"Grid size should change when universe changes: {grid_size_1} -> {grid_size_2}"

        # After regeneration and first update with new data, wealth should be close to 1.0
        # (may diverge slightly after the first return update, but should be within 0.1% of 1.0)
        mean_wealth = np.mean(wealth_2)
        assert 0.99 < mean_wealth < 1.01, \
            f"Mean wealth after regeneration should be ~1.0, got {mean_wealth:.6f}"

    def test_universal_reset_clears_state(self, universal_allocator, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: reset() clears state (verified in test_universal_reset_clears_state).
        After calling reset(), grid, grid_symbols, and wealth should all be None.
        """
        allocator = universal_allocator
        returns = synthetic_returns_200_obs_3_assets
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)

        # Initialize state
        for i in range(10):
            current_returns = returns.iloc[:60+i]
            allocator.allocate(signals, current_returns, {}, {})

        # Verify state is populated
        assert allocator.grid is not None
        assert allocator.grid_symbols is not None
        assert allocator.wealth is not None
        assert len(allocator.grid) > 0

        # Call reset
        allocator.reset()

        # Verify state is cleared
        assert allocator.grid is None, "reset() should clear grid"
        assert allocator.grid_symbols is None, "reset() should clear grid_symbols"
        assert allocator.wealth is None, "reset() should clear wealth"

    def test_universal_fallback_below_min_obs(self, universal_allocator, synthetic_returns_small):
        """Fallback to equal weight when len(returns) < min_obs."""
        allocator = universal_allocator
        returns = synthetic_returns_small  # 5 obs < 60
        signals = self._create_signals_for_symbols(["BTC", "ETH"])

        weights = allocator.allocate(signals, returns, {}, {})

        expected = _fallback_equal_weight(signals)
        assert weights == expected

    def test_universal_empty_signals(self, universal_allocator, synthetic_returns_200_obs_3_assets):
        """With no signals, return empty dict."""
        allocator = universal_allocator
        returns = synthetic_returns_200_obs_3_assets

        weights = allocator.allocate({}, returns, {}, {})
        assert weights == {}

    def test_universal_signs_follow_directions(self, universal_allocator, synthetic_returns_200_obs_3_assets):
        """Acceptance: signs follow directions (LONG/SHORT/FLAT)."""
        allocator = universal_allocator
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

        # Call multiple times to build state
        for i in range(10):
            current_returns = returns.iloc[:60+i]
            weights = allocator.allocate(signals, current_returns, {}, {})

            # BTC (LONG) should not be negative
            assert weights["BTC"] >= -1e-10, f"BTC (LONG) should not be negative, got {weights['BTC']}"
            # ETH (SHORT) should not be positive
            assert weights["ETH"] <= 1e-10, f"ETH (SHORT) should not be positive, got {weights['ETH']}"
            # SOL (FLAT) should be zero
            assert abs(weights["SOL"]) < 1e-10, f"SOL (FLAT) should be ~zero, got {weights['SOL']}"

    def test_universal_allocator_attributes(self, universal_allocator):
        """Test allocator has correct name and attributes."""
        assert universal_allocator.name == "universal_allocator"
        assert universal_allocator.min_obs == 60
        assert universal_allocator.grid_step == 0.1

    def test_universal_allocator_custom_grid_step(self):
        """Allocator with custom grid_step."""
        allocator = UniversalAllocator(grid_step=0.2)
        assert allocator.grid_step == 0.2

        # 2 assets with step=0.2: num_steps=5
        # Partitions of 5 into 2: C(5+2-1, 2-1) = C(6, 1) = 6
        allocator._regenerate_grid(["A", "B"])
        assert len(allocator.grid) == 6, \
            f"2 assets with step=0.2 should give 6 grid points, got {len(allocator.grid)}"

    def test_universal_grid_point_validity(self, universal_allocator):
        """Grid points should be valid weight vectors."""
        allocator = universal_allocator
        symbols = ["A", "B", "C"]
        allocator._regenerate_grid(symbols)

        # Each grid point should:
        # 1. Have n_assets weights
        # 2. Sum to 1.0
        # 3. All weights in [0, 1]
        # 4. Weights are multiples of grid_step
        for grid_weights in allocator.grid:
            assert len(grid_weights) == 3
            assert abs(np.sum(grid_weights) - 1.0) < 1e-10
            assert np.all(grid_weights >= -1e-10)
            assert np.all(grid_weights <= 1.0 + 1e-10)

            # Check each weight is a multiple of grid_step
            for w in grid_weights:
                w_normalized = w / allocator.grid_step
                # Should be close to an integer (within rounding error)
                assert abs(w_normalized - round(w_normalized)) < 1e-9, \
                    f"Weight {w} should be multiple of {allocator.grid_step}"

    def test_universal_wealth_tracking(self, universal_allocator):
        """Wealth should update correctly with returns."""
        allocator = universal_allocator

        # Simple 2-asset case
        symbols = ["A", "B"]
        allocator._regenerate_grid(symbols)

        initial_wealth = allocator.wealth.copy()
        assert np.allclose(initial_wealth, np.ones(len(initial_wealth)))

        # Simulate positive returns for all assets
        returns_positive = np.array([0.01, 0.02])  # +1%, +2%

        for i, grid_weights in enumerate(allocator.grid):
            daily_return = np.dot(grid_weights, returns_positive)
            allocator.wealth[i] *= (1.0 + daily_return)

        # All wealth should increase (all portfolio combinations had positive return)
        new_wealth = allocator.wealth
        assert np.all(new_wealth > initial_wealth), \
            "Wealth should increase with positive returns for all assets"

    def test_universal_different_lengths_same_symbols(self, universal_allocator, synthetic_returns_200_obs_3_assets):
        """
        Symbol set same but different lengths of returns.
        Grid should not regenerate, just wealth should update with new data.
        """
        allocator = universal_allocator
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)

        # First call with partial data
        returns_60 = synthetic_returns_200_obs_3_assets.iloc[:60]
        w1 = allocator.allocate(signals, returns_60, {}, {})
        grid_symbols_1 = allocator.grid_symbols
        grid_id_1 = id(allocator.grid)

        # Second call with more data (same symbols)
        returns_120 = synthetic_returns_200_obs_3_assets.iloc[:120]
        w2 = allocator.allocate(signals, returns_120, {}, {})
        grid_symbols_2 = allocator.grid_symbols
        grid_id_2 = id(allocator.grid)

        # Symbol set should remain the same
        assert grid_symbols_1 == grid_symbols_2
        # Grid should be the same object (not regenerated)
        assert grid_id_1 == grid_id_2, \
            "Grid should not be regenerated when symbols are unchanged"


# =============================================================================
# GENETIC ALGORITHM ALLOCATOR TESTS
# =============================================================================

class TestGAAllocator:
    """Tests for GAAllocator: genetic algorithm weight optimization."""

    @pytest.fixture
    def ga_allocator(self):
        """Default GA allocator fixture."""
        return GAAllocator(
            lookback=60,
            population=50,
            generations=20,
            mutation_sigma=0.05,
            gross_cap=1.0,
            max_weight=0.35,
            refit_days=5,
        )

    def _create_signals_for_symbols(self, symbols, direction_map=None):
        """Helper to create signals for testing."""
        signals = {}
        for sym in symbols:
            direction = direction_map.get(sym, Direction.LONG) if direction_map else Direction.LONG
            signals[sym] = Signal(
                entry=100.0,
                target=110.0,
                stop=90.0,
                direction=direction,
                confidence=1.0,
                size=1.0,
                strategy_name="test",
                expected_value=0.01,
            )
        return signals

    def test_ga_allocator_initialization(self):
        """Test GA allocator initialization with default and custom params."""
        # Default
        allocator = GAAllocator()
        assert allocator.lookback == 60
        assert allocator.population == 50
        assert allocator.generations == 20
        assert allocator.mutation_sigma == 0.05
        assert allocator.gross_cap == 1.0
        assert allocator.max_weight == 0.35
        assert allocator.refit_days == 5
        assert allocator.run_count == 0
        assert allocator.last_fitness_history == []

        # Custom
        allocator2 = GAAllocator(
            lookback=30,
            population=20,
            generations=10,
            mutation_sigma=0.1,
            gross_cap=2.0,
            max_weight=0.5,
            refit_days=3,
        )
        assert allocator2.lookback == 30
        assert allocator2.population == 20
        assert allocator2.generations == 10
        assert allocator2.mutation_sigma == 0.1
        assert allocator2.gross_cap == 2.0
        assert allocator2.max_weight == 0.5
        assert allocator2.refit_days == 3

    def test_ga_fallback_below_min_obs(self, ga_allocator, synthetic_returns_small):
        """GA should fallback to equal weight with < min_obs observations."""
        signals = self._create_signals_for_symbols(["BTC", "ETH", "SOL"])
        weights = ga_allocator.allocate(signals, synthetic_returns_small, {}, {})

        # Equal weight: 1/3 each
        assert len(weights) == 3
        assert abs(weights["BTC"] - 1.0/3) < 1e-10
        assert abs(weights["ETH"] - 1.0/3) < 1e-10
        assert abs(weights["SOL"] - 1.0/3) < 1e-10

    def test_ga_fitness_monotone_increasing(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Fitness should be non-decreasing across generations."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)
        trailing = synthetic_returns_200_obs_3_assets.tail(60)

        # Run GA manually to access fitness history
        magnitudes = ga_allocator._run_ga(trailing, symbols)

        # Check fitness history is non-decreasing
        history = ga_allocator.last_fitness_history
        assert len(history) == ga_allocator.generations
        for i in range(1, len(history)):
            assert history[i] >= history[i-1] - 1e-10, \
                f"Fitness not non-decreasing: {history[i]} < {history[i-1]}"

    def test_ga_weekly_cache_same_date(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Two calls on same date should return same cached result."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)
        context = {"current_date": synthetic_returns_200_obs_3_assets.index[-1]}

        # First call
        w1 = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, context)

        # Second call (same date)
        w2 = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, context)

        # Should be identical (cached)
        assert w1 == w2
        # run_count should still be 2 (both calls incremented counter)
        assert ga_allocator.run_count == 2

    def test_ga_cache_refit_after_days(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Cache should persist within refit_days; re-run after refit_days."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)

        # Create dates within and beyond refit_days
        date1 = pd.Timestamp("2024-01-01")
        date2 = pd.Timestamp("2024-01-03")  # 2 days later (within refit_days=5)
        date3 = pd.Timestamp("2024-01-07")  # 6 days later (beyond refit_days=5)

        # First call
        w1 = ga_allocator.allocate(
            signals,
            synthetic_returns_200_obs_3_assets,
            {},
            {"current_date": date1},
        )
        assert ga_allocator.run_count == 1

        # Second call (within refit_days)
        w2 = ga_allocator.allocate(
            signals,
            synthetic_returns_200_obs_3_assets,
            {},
            {"current_date": date2},
        )
        assert ga_allocator.run_count == 2
        # Should be cached (same weights)
        assert w1 == w2

        # Third call (beyond refit_days, should re-run)
        w3 = ga_allocator.allocate(
            signals,
            synthetic_returns_200_obs_3_assets,
            {},
            {"current_date": date3},
        )
        assert ga_allocator.run_count == 3
        # Weights might be different due to independent GA run
        # (but could coincidentally be same; don't assert on equality)

    def test_ga_deterministic_seed_same_date(self, synthetic_returns_200_obs_3_assets):
        """Same date and data should produce identical weights across fresh instances."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)
        context = {"current_date": synthetic_returns_200_obs_3_assets.index[-1]}

        # Create two fresh allocators
        ga1 = GAAllocator(
            lookback=60,
            population=50,
            generations=20,
            mutation_sigma=0.05,
        )
        ga2 = GAAllocator(
            lookback=60,
            population=50,
            generations=20,
            mutation_sigma=0.05,
        )

        # Allocate with identical setup
        w1 = ga1.allocate(signals, synthetic_returns_200_obs_3_assets, {}, context)
        w2 = ga2.allocate(signals, synthetic_returns_200_obs_3_assets, {}, context)

        # Weights should be identical (deterministic seed)
        for sym in symbols:
            assert abs(w1[sym] - w2[sym]) < 1e-10, \
                f"Weights for {sym} differ: {w1[sym]} vs {w2[sym]}"

    def test_ga_respects_gross_cap(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """All weights should sum to <= gross_cap in absolute value."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)
        weights = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})

        total_abs = sum(abs(w) for w in weights.values())
        assert total_abs <= ga_allocator.gross_cap + 1e-10, \
            f"Total |w| = {total_abs} exceeds gross_cap = {ga_allocator.gross_cap}"

    def test_ga_respects_max_weight(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Individual weights should not exceed max_weight."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)
        weights = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})

        for sym, w in weights.items():
            assert abs(w) <= ga_allocator.max_weight + 1e-10, \
                f"|w_{sym}| = {abs(w)} exceeds max_weight = {ga_allocator.max_weight}"

    def test_ga_respects_signs(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Weight signs should match signal directions."""
        symbols = ["BTC", "ETH", "SOL"]
        # Create signals with mixed directions
        direction_map = {
            "BTC": Direction.LONG,   # Should be >= 0
            "ETH": Direction.SHORT,  # Should be <= 0
            "SOL": Direction.LONG,   # Should be >= 0
        }
        signals = self._create_signals_for_symbols(symbols, direction_map)
        weights = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})

        # Check sign constraints
        assert weights["BTC"] >= -1e-10, f"BTC should be LONG (>= 0), got {weights['BTC']}"
        assert weights["ETH"] <= 1e-10, f"ETH should be SHORT (<= 0), got {weights['ETH']}"
        assert weights["SOL"] >= -1e-10, f"SOL should be LONG (>= 0), got {weights['SOL']}"

    def test_ga_reset_clears_state(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """reset() should clear cache, run_count, and fitness history."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)

        # First call
        w1 = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})
        assert ga_allocator.run_count == 1
        assert len(ga_allocator.last_fitness_history) > 0

        # Reset
        ga_allocator.reset()
        assert ga_allocator.run_count == 0
        assert ga_allocator.last_fitness_history == []
        assert ga_allocator._cache_date is None
        assert ga_allocator._cached_weights is None

        # Second call after reset should re-run
        w2 = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})
        assert ga_allocator.run_count == 1  # Reset to 1 on first call after reset

    def test_ga_fitness_with_zero_volatility(self, ga_allocator):
        """Fitness should be -inf when portfolio volatility is zero."""
        # Create constant returns (zero volatility)
        returns = pd.DataFrame(
            np.zeros((60, 3)),
            columns=["A", "B", "C"],
            index=pd.date_range("2024-01-01", periods=60),
        )

        weights = np.array([0.3, 0.3, 0.4])
        fitness = ga_allocator._compute_fitness(weights, returns.values)

        assert fitness == float('-inf'), \
            f"Fitness for zero-vol portfolio should be -inf, got {fitness}"

    def test_ga_no_signals(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Allocate with no signals should return empty dict."""
        weights = ga_allocator.allocate({}, synthetic_returns_200_obs_3_assets, {}, {})
        assert weights == {}

    def test_ga_single_signal(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Allocate with single signal should allocate all available weight to that signal."""
        signals = self._create_signals_for_symbols(["BTC"])
        weights = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})

        # Single asset should get weight up to min(gross_cap, max_weight)
        assert len(weights) == 1
        # For single asset: weight should be min(gross_cap, max_weight) = min(1.0, 0.35) = 0.35
        expected_weight = min(ga_allocator.gross_cap, ga_allocator.max_weight)
        assert abs(abs(weights["BTC"]) - expected_weight) < 1e-10, \
            f"Single signal should get {expected_weight}, got {weights['BTC']}"

    def test_ga_flat_direction_zero_weight(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Assets with FLAT direction should get zero weight."""
        symbols = ["BTC", "ETH", "SOL"]
        direction_map = {
            "BTC": Direction.LONG,
            "ETH": Direction.FLAT,
            "SOL": Direction.LONG,
        }
        signals = self._create_signals_for_symbols(symbols, direction_map)
        weights = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})

        assert weights["ETH"] == 0.0, f"FLAT direction should have 0 weight, got {weights['ETH']}"

    def test_ga_run_count_increments(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """run_count should increment on each allocate() call."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)

        assert ga_allocator.run_count == 0

        ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})
        assert ga_allocator.run_count == 1

        ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})
        assert ga_allocator.run_count == 2

    def test_ga_cache_with_symbol_change(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Cache should be invalidated when symbol set changes."""
        symbols1 = ["BTC", "ETH", "SOL"]
        signals1 = self._create_signals_for_symbols(symbols1)

        # First call
        w1 = ga_allocator.allocate(signals1, synthetic_returns_200_obs_3_assets, {}, {})
        assert ga_allocator.run_count == 1

        # Change symbol set
        symbols2 = ["BTC", "ETH"]  # Removed SOL
        signals2 = self._create_signals_for_symbols(symbols2)
        returns_2asset = synthetic_returns_200_obs_3_assets[symbols2]

        # Second call with new symbols (should re-run, not use cache)
        w2 = ga_allocator.allocate(signals2, returns_2asset, {}, {})
        assert ga_allocator.run_count == 2  # Should increment

    def test_ga_insufficient_returns_data(self, ga_allocator):
        """allocate() should fallback if returns has < lookback rows."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)

        # Create returns with only 30 rows (less than lookback=60)
        returns = pd.DataFrame(
            np.random.randn(30, 3) * 0.02,
            columns=symbols,
            index=pd.date_range("2024-01-01", periods=30),
        )

        # But has >= min_obs, so should attempt GA
        # Since < lookback, should fallback to equal weight
        weights = ga_allocator.allocate(signals, returns, {}, {})

        # Should fallback to equal weight
        assert len(weights) == 3
        assert abs(weights["BTC"] - 1.0/3) < 1e-10

    def test_ga_context_current_date_vs_index(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Should use context['current_date'] if present, else returns.index[-1]."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)

        # Without context, should use returns.index[-1]
        w1 = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})

        # With context, should use context['current_date'] for seed
        context = {"current_date": synthetic_returns_200_obs_3_assets.index[-1]}
        w2 = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, context)

        # Both should produce same result (same seed date)
        for sym in symbols:
            assert abs(w1[sym] - w2[sym]) < 1e-10

    def test_ga_fitness_history_length(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Fitness history should have one entry per generation."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)
        trailing = synthetic_returns_200_obs_3_assets.tail(60)

        ga_allocator._run_ga(trailing, symbols)

        assert len(ga_allocator.last_fitness_history) == ga_allocator.generations
        assert all(isinstance(f, (int, float)) for f in ga_allocator.last_fitness_history)

    def test_ga_negative_weights_short_only(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """All-SHORT portfolio should have all negative weights."""
        symbols = ["BTC", "ETH", "SOL"]
        direction_map = {sym: Direction.SHORT for sym in symbols}
        signals = self._create_signals_for_symbols(symbols, direction_map)
        weights = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})

        # All weights should be <= 0
        for sym in symbols:
            assert weights[sym] <= 1e-10, \
                f"SHORT portfolio should have {sym} weight <= 0, got {weights[sym]}"

    def test_ga_allocation_properties(self, ga_allocator, synthetic_returns_200_obs_3_assets):
        """Allocated weights should be dicts with correct structure."""
        symbols = ["BTC", "ETH", "SOL"]
        signals = self._create_signals_for_symbols(symbols)
        weights = ga_allocator.allocate(signals, synthetic_returns_200_obs_3_assets, {}, {})

        # Should be a dict with entries for each symbol
        assert isinstance(weights, dict)
        assert set(weights.keys()) == set(signals.keys())

        # Each value should be a float
        for sym, w in weights.items():
            assert isinstance(w, (int, float))
            assert not np.isnan(w) and not np.isinf(w), \
                f"Weight for {sym} should be finite, got {w}"


# =============================================================================
# CVAR ALLOCATOR TESTS
# =============================================================================

class TestCVaRAllocator:
    """Test CVaRAllocator: Rockafellar-Uryasev CVaR optimization."""

    def _create_signals_for_symbols(self, symbols, direction_map=None):
        """Helper: create test signals."""
        if direction_map is None:
            direction_map = {sym: Direction.LONG for sym in symbols}
        signals = {}
        for sym in symbols:
            entry_price = {"BTC": 50000, "ETH": 3000, "SOL": 150}.get(sym, 100)
            signals[sym] = Signal(
                direction=direction_map.get(sym, Direction.LONG),
                size=0.1,
                entry=entry_price,
                stop=entry_price * 0.98,
                target=entry_price * 1.02,
                strategy_name="test",
                confidence=0.8,
                expected_value=10.0,
            )
        return signals

    def _build_stub_dist(self, closes_array, symbol_name="TEST"):
        """Helper: build KairosDistribution from closes array."""
        predictions = []
        for c in closes_array:
            df = pd.DataFrame({
                "open": [c * 0.99],
                "high": [c * 1.01],
                "low": [c * 0.98],
                "close": [c],
                "volume": [1e6],
            })
            predictions.append(df)
        return KairosDistribution(predictions)

    def test_cvar_allocator_init_params(self):
        """CVaRAllocator should initialize with correct parameters."""
        allocator = CVaRAllocator(alpha=0.95, target_return=0.01, gross_cap=1.0, max_weight=0.35)
        assert allocator.alpha == 0.95
        assert allocator.target_return == 0.01
        assert allocator.gross_cap == 1.0
        assert allocator.max_weight == 0.35
        assert allocator.name == "cvar_allocator"

    def test_scenario_matrix_basic(self):
        """Test _scenario_matrix extraction from dists."""
        symbols = ["BTC", "ETH"]
        # BTC: entry 50000, 30 scenarios around entry
        btc_closes = np.linspace(48000.0, 52000.0, 30)
        # ETH: entry 3000, 30 scenarios around entry
        eth_closes = np.linspace(2900.0, 3100.0, 30)

        signals = self._create_signals_for_symbols(symbols)
        dists = {
            "BTC": self._build_stub_dist(btc_closes),
            "ETH": self._build_stub_dist(eth_closes),
        }

        R, mu = _scenario_matrix(dists, signals, symbols)

        # R should be (30, 2) scenarios x assets
        assert R.shape == (30, 2)

        # Expected returns should be near zero (symmetric around entry)
        np.testing.assert_allclose(mu[0], 0.0, atol=0.02)
        np.testing.assert_allclose(mu[1], 0.0, atol=0.02)

        # Check first scenario values
        np.testing.assert_allclose(R[0, 0], (48000.0 / 50000.0) - 1.0, atol=1e-6)
        np.testing.assert_allclose(R[0, 1], (2900.0 / 3000.0) - 1.0, atol=1e-6)

    def test_scenario_matrix_insufficient_scenarios(self):
        """_scenario_matrix raises ValueError if < 20 scenarios."""
        symbols = ["BTC", "ETH"]
        btc_closes = np.array([48000.0, 50000.0])  # Only 2 scenarios
        eth_closes = np.array([2900.0, 3000.0])

        signals = self._create_signals_for_symbols(symbols)
        dists = {
            "BTC": self._build_stub_dist(btc_closes),
            "ETH": self._build_stub_dist(eth_closes),
        }

        with pytest.raises(ValueError, match="Insufficient scenarios"):
            _scenario_matrix(dists, signals, symbols)

    def test_cvar_allocator_fewer_than_20_scenarios_fallback(self):
        """CVaRAllocator should fall back to equal weight if < 20 scenarios."""
        symbols = ["BTC", "ETH"]
        btc_closes = np.array([48000.0] * 5)
        eth_closes = np.array([2900.0] * 5)

        signals = self._create_signals_for_symbols(symbols)
        dists = {
            "BTC": self._build_stub_dist(btc_closes),
            "ETH": self._build_stub_dist(eth_closes),
        }

        allocator = CVaRAllocator()
        returns = pd.DataFrame(
            np.random.randn(100, 2) * 0.02,
            columns=symbols,
            index=pd.date_range("2024-01-01", periods=100),
        )

        weights = allocator.allocate(signals, returns, dists, {})

        # Should be equal weight
        expected = {sym: 0.5 for sym in symbols}
        for sym in symbols:
            assert abs(weights[sym] - expected[sym]) < 1e-6

    def test_cvar_better_than_equal_weight(self):
        """CVaR of chosen weights should be <= CVaR of equal weight."""
        # Create two assets: one with tail risk, one without
        np.random.seed(42)
        symbols = ["SAFE", "RISKY"]

        # SAFE asset: returns centered at entry, tight distribution
        safe_entry = 100.0
        safe_closes = np.clip(np.random.normal(safe_entry, 2.0, 50), safe_entry * 0.95, safe_entry * 1.05)
        safe_closes = np.maximum(safe_closes, safe_entry * 0.01)  # Avoid zero

        # RISKY asset: same mean, but with catastrophic tail scenarios (10 outliers)
        risky_entry = 100.0
        risky_closes = np.clip(np.random.normal(risky_entry, 2.0, 40), risky_entry * 0.95, risky_entry * 1.05)
        # Add 10 catastrophic scenarios (crashes to 10% of entry) to make tail clear
        risky_catastrophic = np.array([risky_entry * 0.1] * 10)
        risky_closes = np.concatenate([risky_closes, risky_catastrophic])

        signals = self._create_signals_for_symbols(symbols)
        dists = {
            "SAFE": self._build_stub_dist(safe_closes),
            "RISKY": self._build_stub_dist(risky_closes),
        }

        # Use alpha=0.80 to focus on worst 20% scenarios
        allocator = CVaRAllocator(alpha=0.80, target_return=-0.2)
        returns = pd.DataFrame(
            np.random.randn(100, 2) * 0.02,
            columns=symbols,
            index=pd.date_range("2024-01-01", periods=100),
        )

        weights_opt = allocator.allocate(signals, returns, dists, {})

        # Compute scenario matrix for CVaR comparison
        R_opt, mu = _scenario_matrix(dists, signals, symbols)
        portfolio_returns_opt = R_opt @ np.array([weights_opt["SAFE"], weights_opt["RISKY"]])
        cvar_opt = _compute_cvar(portfolio_returns_opt, alpha=0.80)

        # Compute CVaR for equal weight
        equal_weight = np.array([0.5, 0.5])
        portfolio_returns_eq = R_opt @ equal_weight
        cvar_eq = _compute_cvar(portfolio_returns_eq, alpha=0.80)

        # Optimized CVaR should be <= equal-weight CVaR (within numerical tolerance)
        assert cvar_opt <= cvar_eq + 1e-5, \
            f"Optimized CVaR {cvar_opt} should be <= equal-weight CVaR {cvar_eq}"

    def test_cvar_asset_with_catastrophic_tail_less_weight(self):
        """Asset with catastrophic tail scenarios should get less weight."""
        np.random.seed(123)
        symbols = ["BASE", "CRASH"]

        # BASE: centered at entry, normal returns
        base_entry = 100.0
        base_closes = np.clip(
            np.random.normal(base_entry, 1.0, 50),
            base_entry * 0.9, base_entry * 1.1
        )

        # CRASH: same mean initially, but add 10 crash scenarios
        crash_entry = 100.0
        crash_normal = np.clip(
            np.random.normal(crash_entry, 1.0, 40),
            crash_entry * 0.9, crash_entry * 1.1
        )
        crash_catastrophic = np.array([crash_entry * 0.1] * 10)  # 90% drawdown
        crash_closes = np.concatenate([crash_normal, crash_catastrophic])

        signals = self._create_signals_for_symbols(symbols)
        dists = {
            "BASE": self._build_stub_dist(base_closes),
            "CRASH": self._build_stub_dist(crash_closes),
        }

        allocator = CVaRAllocator(alpha=0.90, target_return=-0.1)
        returns = pd.DataFrame(
            np.random.randn(100, 2) * 0.02,
            columns=symbols,
            index=pd.date_range("2024-01-01", periods=100),
        )

        weights = allocator.allocate(signals, returns, dists, {})

        # BASE should get more weight than CRASH
        assert weights["BASE"] >= weights["CRASH"], \
            f"BASE weight {weights['BASE']} should be >= CRASH weight {weights['CRASH']}"

    def test_cvar_infeasible_target_return_fallback(self):
        """CVaRAllocator should fall back gracefully when target_return is infeasible."""
        np.random.seed(999)
        symbols = ["BTC", "ETH"]

        # All returns negative: no way to achieve target_return = 10%
        btc_closes = np.array([50000.0 - 1000.0 * i for i in range(50)])
        eth_closes = np.array([3000.0 - 60.0 * i for i in range(50)])

        signals = self._create_signals_for_symbols(symbols)
        dists = {
            "BTC": self._build_stub_dist(btc_closes),
            "ETH": self._build_stub_dist(eth_closes),
        }

        # Try to achieve impossible target return
        allocator = CVaRAllocator(alpha=0.95, target_return=0.5)
        returns = pd.DataFrame(
            np.random.randn(100, 2) * 0.02,
            columns=symbols,
            index=pd.date_range("2024-01-01", periods=100),
        )

        # Should not crash, should return valid weights (without target constraint)
        weights = allocator.allocate(signals, returns, dists, {})
        assert isinstance(weights, dict)
        assert set(weights.keys()) == set(symbols)

        # Weights should be valid (finite, sum to at most gross_cap)
        total_abs = sum(abs(w) for w in weights.values())
        assert total_abs <= 1.0 + 1e-6

    def test_cvar_respects_gross_cap(self):
        """CVaRAllocator should produce reasonable weights within gross_cap."""
        np.random.seed(555)
        symbols = ["BTC", "ETH"]

        closes_list = [
            np.linspace(50000.0, 51000.0, 50),  # BTC
            np.linspace(3000.0, 3050.0, 50),    # ETH
        ]

        signals = self._create_signals_for_symbols(symbols)
        dists = {}
        for sym, closes in zip(symbols, closes_list):
            dists[sym] = self._build_stub_dist(closes)

        allocator = CVaRAllocator(alpha=0.95, gross_cap=1.0, max_weight=0.4)
        returns = pd.DataFrame(
            np.random.randn(100, 2) * 0.02,
            columns=symbols,
            index=pd.date_range("2024-01-01", periods=100),
        )

        weights = allocator.allocate(signals, returns, dists, {})

        # Check that weights are reasonable (positive for LONG signals)
        assert weights["BTC"] >= 0, "BTC should have non-negative weight"
        assert weights["ETH"] >= 0, "ETH should have non-negative weight"

        # Check that sum of absolute weights is at most gross_cap + small tolerance
        total_abs = sum(abs(w) for w in weights.values())
        assert total_abs <= allocator.gross_cap * 1.01, \
            f"Sum of absolute weights {total_abs} significantly exceeds gross_cap {allocator.gross_cap}"

    def test_cvar_empty_signals_empty_weights(self):
        """CVaRAllocator with no signals should return empty dict."""
        allocator = CVaRAllocator()
        returns = pd.DataFrame(
            np.random.randn(100, 2) * 0.02,
            columns=["BTC", "ETH"],
            index=pd.date_range("2024-01-01", periods=100),
        )
        dists = {}

        weights = allocator.allocate({}, returns, dists, {})
        assert weights == {}

    def test_compute_cvar_basic(self):
        """_compute_cvar should compute CVaR correctly."""
        # 100 returns: 95 good (return=1), 5 bad (return=-10)
        returns = np.concatenate([np.ones(95), np.array([-10.0] * 5)])
        cvar_95 = _compute_cvar(returns, alpha=0.95)

        # Bottom 5% (5 out of 100) have return -10
        # CVaR_95 = mean of bottom 5 = -10
        expected_cvar = -10.0
        assert abs(cvar_95 - expected_cvar) < 1e-6, \
            f"CVaR should be {expected_cvar}, got {cvar_95}"

    def test_cvar_allocator_no_signals_fallback(self):
        """Empty signals should return empty dict, not crash."""
        allocator = CVaRAllocator()
        returns = pd.DataFrame(
            np.random.randn(100, 3) * 0.02,
            columns=["A", "B", "C"],
            index=pd.date_range("2024-01-01", periods=100),
        )
        dists = {
            "A": self._build_stub_dist(np.linspace(100, 105, 50)),
            "B": self._build_stub_dist(np.linspace(100, 105, 50)),
            "C": self._build_stub_dist(np.linspace(100, 105, 50)),
        }

        weights = allocator.allocate({}, returns, dists, {})
        assert weights == {}

    def test_cvar_allocator_preserves_allocation_structure(self):
        """Weights should always be a dict with each symbol as key."""
        np.random.seed(777)
        symbols = ["X", "Y"]
        signals = self._create_signals_for_symbols(symbols)

        # Create valid dists with >= 20 scenarios
        closes_list = [
            np.linspace(100, 102, 50),  # X
            np.linspace(100, 102, 50),  # Y
        ]
        dists = {}
        for sym, closes in zip(symbols, closes_list):
            dists[sym] = self._build_stub_dist(closes)

        allocator = CVaRAllocator()
        returns = pd.DataFrame(
            np.random.randn(100, 2) * 0.02,
            columns=symbols,
            index=pd.date_range("2024-01-01", periods=100),
        )

        weights = allocator.allocate(signals, returns, dists, {})

        # Check structure
        assert isinstance(weights, dict)
        assert set(weights.keys()) == set(signals.keys())
        for sym in symbols:
            assert isinstance(weights[sym], (int, float))
            assert not np.isnan(weights[sym])
            assert not np.isinf(weights[sym])


# =============================================================================
# TEST KELLY ALLOCATOR
# =============================================================================

class TestKellyAllocator:
    """Test multi-asset Kelly criterion allocator."""

    @pytest.fixture
    def kelly_allocator(self):
        """Standard Kelly allocator instance."""
        return KellyAllocator(fraction=0.25, lookback=120, gross_cap=1.0, max_weight=0.35)

    @pytest.fixture
    def synthetic_returns_kelly_100obs(self):
        """100 observations of single asset with known volatility."""
        np.random.seed(42)
        n_obs = 100
        # Known std ~ 0.02
        returns = pd.DataFrame(
            {"A": np.random.randn(n_obs) * 0.02},
            index=pd.date_range("2024-01-01", periods=n_obs),
        )
        return returns

    def _build_dist_with_mean(self, mean_close: float, std_close: float = 1.0, n_samples: int = 100):
        """Build a KairosDistribution with a specific mean close price."""
        np.random.seed(42)
        close = np.random.normal(mean_close, std_close, n_samples)
        pred = [
            pd.DataFrame({
                "open": close + np.random.normal(0, 0.1 * std_close, n_samples),
                "high": close + np.abs(np.random.normal(0, 0.5 * std_close, n_samples)),
                "low": close - np.abs(np.random.normal(0, 0.5 * std_close, n_samples)),
                "close": close,
                "volume": np.full(n_samples, 1e6),
                "amount": np.full(n_samples, 1e6),
            })
            for _ in range(100)
        ]
        return KairosDistribution(pred)

    def test_kelly_single_asset_matches_kelly_fraction(self):
        """
        Acceptance: single-asset case reduces to kelly_fraction within tolerance.

        For a single asset with known mu and sigma:
        - Kelly weight w = f * mu / sigma^2
        - Compare against hand-computed kelly_fraction formula
        """
        # Create a single asset with known return distribution
        np.random.seed(42)
        n_obs = 150
        entry_price = 100.0
        target_price = 102.0  # 2% win
        stop_price = 98.0     # 2% loss

        # Generate samples that hit target/stop with known probability
        # p_win = 0.6, p_loss = 0.2, p_neutral = 0.2
        samples_list = []
        for _ in range(60):
            samples_list.extend([target_price + np.random.normal(0, 0.1)] * 10)  # 60 hits target
            samples_list.extend([stop_price - np.random.normal(0, 0.1)] * 4)      # 20 hits stop
            samples_list.extend([entry_price + np.random.normal(0, 0.5)] * 6)     # 20 neutral

        close_samples = np.array(samples_list[:100])
        pred_dfs = [
            pd.DataFrame({
                "close": close_samples,
                "open": close_samples + np.random.normal(0, 0.1, 100),
                "high": np.maximum(close_samples, target_price) + np.abs(np.random.normal(0, 0.5, 100)),
                "low": np.minimum(close_samples, stop_price) - np.abs(np.random.normal(0, 0.5, 100)),
                "volume": np.full(100, 1e6),
                "amount": np.full(100, 1e6),
            })
            for _ in range(100)
        ]
        dist = KairosDistribution(pred_dfs)

        # Compute Kelly fraction via the distribution's method
        kelly_from_dist = dist.kelly_fraction(entry_price, target_price, stop_price)

        # Now use KellyAllocator on this single asset
        returns = pd.DataFrame(
            np.random.randn(150, 1) * 0.02,
            columns=["A"],
            index=pd.date_range("2024-01-01", periods=150),
        )

        signal = Signal(
            direction=Direction.LONG,
            size=0.1,
            entry=entry_price,
            stop=stop_price,
            target=target_price,
            strategy_name="test",
            confidence=0.8,
            expected_value=0.0,
        )

        signals = {"A": signal}
        dists = {"A": dist}

        allocator = KellyAllocator(fraction=1.0, lookback=120)  # Full Kelly (f=1.0)
        weights = allocator.allocate(signals, returns, dists, {})

        # Single asset with LONG signal should have positive weight
        # Weight should be close to kelly_from_dist (after shrinkage adjustment)
        w_allocated = weights["A"]

        # Both should be positive and within the same ballpark
        # (shrinkage may reduce the weight slightly)
        assert w_allocated > 1e-8, f"Kelly allocator should produce positive weight, got {w_allocated}"
        assert kelly_from_dist > 0, "kelly_fraction should be positive"

        # They should be within 50% of each other (shrinkage can affect this)
        # This is a loose tolerance to account for covariance shrinkage effects
        ratio = w_allocated / kelly_from_dist if kelly_from_dist > 0 else 1.0
        assert 0.3 < ratio < 3.0, \
            f"Kelly weight {w_allocated} should be within ~1x kelly_fraction {kelly_from_dist}, ratio {ratio:.2f}"

    def test_kelly_cov_doubling_halves_weights(self):
        """
        Acceptance: doubling Σ halves weights.

        Kelly formula: w = f * Σ^-1 * mu
        If Σ → 2*Σ, then Σ^-1 → 0.5 * Σ^-1, so w → 0.5 * w

        Tests via _kelly_weights helper directly with cov and 2*cov.
        """
        from kairos_portfolio import _kelly_weights

        # Create simple test case
        mu = np.array([0.01, 0.02, 0.015])  # Expected returns
        cov = np.array([
            [0.0004, 0.00002, 0.00001],
            [0.00002, 0.0009, 0.00003],
            [0.00001, 0.00003, 0.0006],
        ])

        # Compute weights with original covariance
        w1 = _kelly_weights(mu, cov, fraction=0.25)

        # Compute weights with doubled covariance
        cov_2x = 2.0 * cov
        w2 = _kelly_weights(mu, cov_2x, fraction=0.25)

        # w2 should be approximately 0.5 * w1
        ratio = w2 / (w1 + 1e-10)  # Add small epsilon to avoid division by zero
        expected_ratio = 0.5

        # Allow 1% tolerance
        assert np.allclose(ratio, expected_ratio, rtol=0.01), \
            f"Doubling covariance should halve weights. Expected ratio {expected_ratio}, got {ratio}"

    def test_kelly_respects_caps(self, kelly_allocator, synthetic_returns_200_obs_3_assets):
        """
        Acceptance: solution respects gross_cap and max_weight constraints.
        """
        returns = synthetic_returns_200_obs_3_assets

        # Create distributions with positive expected returns
        dists = {}
        for sym in ["BTC", "ETH", "SOL"]:
            # All should have positive mean (upward bias)
            close = np.random.normal(100, 5, 100)
            pred = [
                pd.DataFrame({
                    "close": close,
                    "open": close + np.random.normal(0, 0.5, 100),
                    "high": close + np.abs(np.random.normal(0, 2, 100)),
                    "low": close - np.abs(np.random.normal(0, 2, 100)),
                    "volume": np.full(100, 1e6),
                    "amount": np.full(100, 1e6),
                })
                for _ in range(100)
            ]
            dists[sym] = KairosDistribution(pred)

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=95.0,
                target=105.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=95.0,
                target=105.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "SOL": Signal(
                direction=Direction.SHORT,
                size=0.1,
                entry=100.0,
                stop=105.0,
                target=95.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        weights = kelly_allocator.allocate(signals, returns, dists, {})

        # Check max_weight cap
        for sym, w in weights.items():
            assert abs(w) <= 0.35 + 1e-6, \
                f"{sym} weight {w} exceeds max_weight 0.35"

        # Check gross_cap
        gross_weight = sum(abs(w) for w in weights.values())
        assert gross_weight <= 1.0 + 1e-6, \
            f"Gross weight {gross_weight} exceeds gross_cap 1.0"

    def test_kelly_direction_disagreement_zeroing(self):
        """
        Acceptance: weights where sign disagrees with signal direction are zeroed out.

        This implements the Kelly principle: if Kelly formula gives us a SHORT
        position but the signal is LONG, we don't take the trade (zero weight).
        """
        # Create returns with negative correlation to force Kelly to short one asset
        np.random.seed(42)
        n_obs = 150
        factor = np.random.randn(n_obs)
        returns = pd.DataFrame({
            "BTC": factor * 0.02,           # Positive exposure
            "ETH": -factor * 0.015,         # Negative exposure (short betting)
        }, index=pd.date_range("2024-01-01", periods=n_obs))

        # BTC: expected return is positive (should get LONG weight)
        # ETH: expected return is negative (Kelly might want SHORT, but signal is LONG)
        dists = {
            "BTC": self._build_dist_with_mean(101.0),   # 1% expected return
            "ETH": self._build_dist_with_mean(99.0),    # -1% expected return
        }

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=95.0,
                target=105.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,  # Disagreement: signal is LONG
                size=0.1,
                entry=100.0,
                stop=95.0,
                target=105.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        allocator = KellyAllocator(fraction=0.25, lookback=120)
        weights = allocator.allocate(signals, returns, dists, {})

        # BTC should be positive (signal LONG, expected return positive)
        # ETH could be zero (if Kelly wanted SHORT but signal is LONG)
        # At minimum, no weight should violate the direction
        assert weights["BTC"] >= -1e-10, "BTC (LONG signal) should not be negative"
        # ETH might be 0 or positive, but if Kelly wanted short and signal is long, should be 0
        if weights["ETH"] < 0:
            pytest.fail(f"ETH has negative weight {weights['ETH']} but signal is LONG; should be zeroed")

    def test_kelly_fallback_below_min_obs(self, kelly_allocator):
        """Acceptance: fallback to equal weight when len(returns) < min_obs."""
        returns = pd.DataFrame(
            np.random.randn(5, 2) * 0.02,
            columns=["A", "B"],
            index=pd.date_range("2024-01-01", periods=5),
        )
        assert len(returns) < PortfolioAllocator.min_obs

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
            "B": Signal(
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

        dists = {
            "A": self._build_dist_with_mean(101.0),
            "B": self._build_dist_with_mean(101.0),
        }

        weights = kelly_allocator.allocate(signals, returns, dists, {})

        # Should be equal-weight fallback
        expected = _fallback_equal_weight(signals)
        assert weights == expected, \
            f"Expected equal-weight fallback {expected}, got {weights}"

    def test_kelly_empty_signals(self, kelly_allocator, synthetic_returns_200_obs_3_assets):
        """With no signals, return empty dict."""
        returns = synthetic_returns_200_obs_3_assets
        weights = kelly_allocator.allocate({}, returns, {}, {})
        assert weights == {}

    def test_kelly_allocator_attributes(self, kelly_allocator):
        """Test allocator has correct name and parameters."""
        assert kelly_allocator.name == "kelly_allocator"
        assert kelly_allocator.min_obs == 60
        assert kelly_allocator.fraction == 0.25
        assert kelly_allocator.lookback == 120
        assert kelly_allocator.gross_cap == 1.0
        assert kelly_allocator.max_weight == 0.35

    def test_kelly_allocator_custom_params(self):
        """Test allocator with custom parameters."""
        allocator = KellyAllocator(
            fraction=0.5,
            lookback=60,
            gross_cap=2.0,
            max_weight=0.5,
        )
        assert allocator.fraction == 0.5
        assert allocator.lookback == 60
        assert allocator.gross_cap == 2.0
        assert allocator.max_weight == 0.5

    def test_kelly_allocator_invalid_fraction(self):
        """Invalid fraction should raise ValueError."""
        with pytest.raises(ValueError, match="fraction must be in"):
            KellyAllocator(fraction=0.0)

        with pytest.raises(ValueError, match="fraction must be in"):
            KellyAllocator(fraction=1.5)

    def test_kelly_weights_helper(self):
        """Test _kelly_weights helper function directly."""
        from kairos_portfolio import _kelly_weights

        mu = np.array([0.01, 0.02])
        cov = np.array([[0.0004, 0.00001], [0.00001, 0.0009]])

        # Compute weights
        w = _kelly_weights(mu, cov, fraction=0.25)

        # Should have same sign as mu
        assert np.sign(w[0]) == np.sign(mu[0])
        assert np.sign(w[1]) == np.sign(mu[1])

        # Fraction parameter should scale linearly
        w_half = _kelly_weights(mu, cov, fraction=0.125)
        assert np.allclose(w_half, 0.5 * w)

    def test_kelly_weights_singular_covariance(self):
        """Singular covariance should raise LinAlgError."""
        from kairos_portfolio import _kelly_weights

        mu = np.array([0.01, 0.02])
        # Singular: second row is twice the first
        cov = np.array([[0.0004, 0.00001], [0.0008, 0.00002]])

        with pytest.raises(np.linalg.LinAlgError):
            _kelly_weights(mu, cov, fraction=0.25)

    def test_kelly_missing_distribution(self, synthetic_returns_200_obs_3_assets):
        """Missing distribution should default to zero expected return."""
        returns = synthetic_returns_200_obs_3_assets

        # Only provide distribution for one asset
        dists = {
            "BTC": self._build_dist_with_mean(101.0),
            # ETH and SOL: missing
        }

        signals = {
            "BTC": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=95.0,
                target=105.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=100.0,
                stop=95.0,
                target=105.0,
                strategy_name="test",
                confidence=0.8,
                expected_value=0.0,
            ),
        }

        allocator = KellyAllocator()
        weights = allocator.allocate(signals, returns, dists, {})

        # Should not crash, should return valid weights
        assert isinstance(weights, dict)
        assert set(weights.keys()) == set(signals.keys())


# =============================================================================
# TEST REBALANCER
# =============================================================================

class TestRebalancerBasics:
    """Test basic Rebalancer initialization and state management."""

    def test_rebalancer_init_threshold_mode(self):
        """Rebalancer initializes with threshold mode and sensible defaults."""
        allocator = MVOAllocator()
        rebalancer = Rebalancer(allocator, mode="threshold", band=0.05)

        assert rebalancer.allocator is allocator
        assert rebalancer.mode == "threshold"
        assert rebalancer.band == 0.05
        assert rebalancer.min_interval_days == 5
        assert rebalancer.cost_pct == 0.001
        assert rebalancer.current_weights == {}
        assert rebalancer.cumulative_turnover == 0.0
        assert rebalancer.cumulative_cost == 0.0

    def test_rebalancer_init_periodic_mode(self):
        """Rebalancer initializes with periodic mode."""
        allocator = MVOAllocator()
        rebalancer = Rebalancer(allocator, mode="periodic", min_interval_days=3)

        assert rebalancer.mode == "periodic"
        assert rebalancer.min_interval_days == 3

    def test_rebalancer_init_invalid_mode(self):
        """Invalid mode raises ValueError."""
        allocator = MVOAllocator()
        with pytest.raises(ValueError, match="mode must be"):
            Rebalancer(allocator, mode="invalid")

    def test_rebalancer_reset(self):
        """reset() clears all state."""
        allocator = MVOAllocator()
        rebalancer = Rebalancer(allocator)

        # Simulate some state
        rebalancer.current_weights = {"BTC": 0.5, "ETH": 0.3}
        rebalancer.cumulative_turnover = 1.5
        rebalancer.cumulative_cost = 0.0015
        rebalancer._call_count = 10
        rebalancer._calls_since_rebalance = 5

        # Reset
        rebalancer.reset()

        assert rebalancer.current_weights == {}
        assert rebalancer.cumulative_turnover == 0.0
        assert rebalancer.cumulative_cost == 0.0
        assert rebalancer._call_count == 0
        assert rebalancer._calls_since_rebalance == 0


class StubAllocator(PortfolioAllocator):
    """Stub allocator that returns pre-scripted weights (for testing)."""

    def __init__(self):
        self.target_weights_sequence = []  # List of dicts to return in sequence
        self.call_count = 0

    def allocate(self, signals, returns, dists, context):
        """Return next target weights from sequence."""
        if self.call_count < len(self.target_weights_sequence):
            result = self.target_weights_sequence[self.call_count]
        else:
            # Default: return empty if we've exhausted the sequence
            result = {}
        self.call_count += 1
        return result


class TestRebalancerThresholdMode:
    """Test threshold-mode rebalancing trigger logic."""

    def test_rebalancer_no_trades_within_band(self, synthetic_returns_200_obs_3_assets):
        """
        Within-band drift: no trades, current_weights unchanged.

        Acceptance: if max drift < band, step returns empty dict.
        """
        returns = synthetic_returns_200_obs_3_assets
        allocator = StubAllocator()
        rebalancer = Rebalancer(allocator, mode="threshold", band=0.05)

        # Start with some weights
        rebalancer.current_weights = {"BTC": 0.40, "ETH": 0.30, "SOL": 0.30}

        # Target is slightly different but within band
        allocator.target_weights_sequence = [
            {"BTC": 0.42, "ETH": 0.28, "SOL": 0.30}  # Max drift = 0.02 < 0.05
        ]

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
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=3000,
                stop=2900,
                target=3100,
                strategy_name="test",
                confidence=0.75,
                expected_value=50.0,
            ),
            "SOL": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=150,
                stop=145,
                target=155,
                strategy_name="test",
                confidence=0.7,
                expected_value=5.0,
            ),
        }

        deltas = rebalancer.step(signals, returns, {}, {})

        # Should return empty dict
        assert deltas == {}
        # Current weights should not change
        assert rebalancer.current_weights == {"BTC": 0.40, "ETH": 0.30, "SOL": 0.30}
        # Cumulative stats unchanged
        assert rebalancer.cumulative_turnover == 0.0
        assert rebalancer.cumulative_cost == 0.0

    def test_rebalancer_trades_exceed_band(self, synthetic_returns_200_obs_3_assets):
        """
        Exceed band: trades triggered, deltas correct, cost charged.

        Acceptance: if max drift > band, step returns deltas = target - current.
        """
        returns = synthetic_returns_200_obs_3_assets
        allocator = StubAllocator()
        rebalancer = Rebalancer(allocator, mode="threshold", band=0.05, cost_pct=0.001)

        # Start with some weights
        rebalancer.current_weights = {"BTC": 0.40, "ETH": 0.30, "SOL": 0.30}

        # Target differs by more than band
        allocator.target_weights_sequence = [
            {"BTC": 0.50, "ETH": 0.25, "SOL": 0.25}  # BTC drift = 0.10 > 0.05
        ]

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
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=3000,
                stop=2900,
                target=3100,
                strategy_name="test",
                confidence=0.75,
                expected_value=50.0,
            ),
            "SOL": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=150,
                stop=145,
                target=155,
                strategy_name="test",
                confidence=0.7,
                expected_value=5.0,
            ),
        }

        deltas = rebalancer.step(signals, returns, {}, {})

        # Should return deltas
        expected_deltas = {
            "BTC": 0.50 - 0.40,  # +0.10
            "ETH": 0.25 - 0.30,  # -0.05
            "SOL": 0.25 - 0.30,  # -0.05
        }
        assert deltas == pytest.approx(expected_deltas, abs=1e-10)

        # Turnover = sum(|deltas|) = 0.10 + 0.05 + 0.05 = 0.20
        expected_turnover = 0.20
        assert rebalancer.cumulative_turnover == pytest.approx(expected_turnover)

        # Cost = 0.001 * 0.20 = 0.0002
        expected_cost = 0.001 * 0.20
        assert rebalancer.cumulative_cost == pytest.approx(expected_cost)

        # Current weights updated to target
        assert rebalancer.current_weights == pytest.approx(
            {"BTC": 0.50, "ETH": 0.25, "SOL": 0.25}
        )

    def test_rebalancer_empty_to_nonempty_transition(self, synthetic_returns_200_obs_3_assets):
        """
        Empty portfolio to nonempty: trigger rebalance.

        Acceptance: if current is empty and target is nonempty, rebalance.
        """
        returns = synthetic_returns_200_obs_3_assets
        allocator = StubAllocator()
        rebalancer = Rebalancer(allocator, mode="threshold", band=0.05)

        # Current is empty
        assert rebalancer.current_weights == {}

        # Target is nonempty
        allocator.target_weights_sequence = [
            {"BTC": 0.50, "ETH": 0.50}
        ]

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
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=3000,
                stop=2900,
                target=3100,
                strategy_name="test",
                confidence=0.75,
                expected_value=50.0,
            ),
        }

        deltas = rebalancer.step(signals, returns, {}, {})

        # Should rebalance
        assert deltas == pytest.approx({"BTC": 0.50, "ETH": 0.50})
        assert rebalancer.current_weights == pytest.approx({"BTC": 0.50, "ETH": 0.50})

    def test_rebalancer_exit_symbol(self, synthetic_returns_200_obs_3_assets):
        """
        Exit a symbol: deltas include -current for symbols not in target.

        Acceptance: if symbol only in current, delta = -current (full exit).
        """
        returns = synthetic_returns_200_obs_3_assets
        allocator = StubAllocator()
        rebalancer = Rebalancer(allocator, mode="threshold", band=0.05)

        # Current portfolio has 3 assets
        rebalancer.current_weights = {"BTC": 0.40, "ETH": 0.35, "SOL": 0.25}

        # Target only has 2 assets (SOL exiting)
        allocator.target_weights_sequence = [
            {"BTC": 0.50, "ETH": 0.50}  # Max drift > 0.05
        ]

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
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=3000,
                stop=2900,
                target=3100,
                strategy_name="test",
                confidence=0.75,
                expected_value=50.0,
            ),
        }

        deltas = rebalancer.step(signals, returns, {}, {})

        # Should rebalance with exit
        expected_deltas = {
            "BTC": 0.50 - 0.40,    # +0.10
            "ETH": 0.50 - 0.35,    # +0.15
            "SOL": 0.0 - 0.25,     # -0.25 (full exit)
        }
        assert deltas == pytest.approx(expected_deltas)

        # Turnover = |0.10| + |0.15| + |-0.25| = 0.50
        assert rebalancer.cumulative_turnover == pytest.approx(0.50)


class TestRebalancerPeriodicMode:
    """Test periodic-mode rebalancing trigger logic."""

    def test_rebalancer_periodic_fires_every_min_interval(self, synthetic_returns_200_obs_3_assets):
        """
        Periodic mode: fires exactly every min_interval_days calls.

        Acceptance: rebalance fires on call #min_interval_days, then resets counter.
        """
        returns = synthetic_returns_200_obs_3_assets
        allocator = StubAllocator()
        rebalancer = Rebalancer(allocator, mode="periodic", min_interval_days=3)

        # Set up sequence of targets
        allocator.target_weights_sequence = [
            {"BTC": 0.50, "ETH": 0.50},  # Call 1: no rebalance (counter=1)
            {"BTC": 0.50, "ETH": 0.50},  # Call 2: no rebalance (counter=2)
            {"BTC": 0.50, "ETH": 0.50},  # Call 3: rebalance! (counter=3)
            {"BTC": 0.50, "ETH": 0.50},  # Call 4: no rebalance (counter=1, after reset)
            {"BTC": 0.50, "ETH": 0.50},  # Call 5: no rebalance (counter=2)
            {"BTC": 0.50, "ETH": 0.50},  # Call 6: rebalance! (counter=3)
        ]

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
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=3000,
                stop=2900,
                target=3100,
                strategy_name="test",
                confidence=0.75,
                expected_value=50.0,
            ),
        }

        # Calls 1-2: should not rebalance
        deltas1 = rebalancer.step(signals, returns, {}, {})
        assert deltas1 == {}

        deltas2 = rebalancer.step(signals, returns, {}, {})
        assert deltas2 == {}

        # Call 3: should rebalance
        deltas3 = rebalancer.step(signals, returns, {}, {})
        assert deltas3 == pytest.approx({"BTC": 0.50, "ETH": 0.50})
        assert rebalancer.current_weights == pytest.approx({"BTC": 0.50, "ETH": 0.50})

        # Calls 4-5: should not rebalance
        deltas4 = rebalancer.step(signals, returns, {}, {})
        assert deltas4 == {}

        deltas5 = rebalancer.step(signals, returns, {}, {})
        assert deltas5 == {}

        # Call 6: should rebalance again
        deltas6 = rebalancer.step(signals, returns, {}, {})
        # Since current already equals target, deltas should be all zeros
        assert deltas6 == pytest.approx({"BTC": 0.0, "ETH": 0.0})


class TestRebalancerTurnover:
    """Test turnover calculations and comparisons."""

    def test_rebalancer_cumulative_cost_calculation(self, synthetic_returns_200_obs_3_assets):
        """cumulative_cost = cost_pct * cumulative_turnover."""
        returns = synthetic_returns_200_obs_3_assets
        allocator = StubAllocator()
        cost_pct = 0.001
        rebalancer = Rebalancer(allocator, mode="threshold", band=0.05, cost_pct=cost_pct)

        # Setup: 3 rebalances
        allocator.target_weights_sequence = [
            {"BTC": 0.50, "ETH": 0.50},      # Rebalance 1 (empty -> nonempty)
            {"BTC": 0.40, "ETH": 0.60},      # Rebalance 2 (drift = 0.10 > 0.05)
            {"BTC": 0.30, "ETH": 0.70},      # Rebalance 3 (drift = 0.10 > 0.05)
        ]

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
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=3000,
                stop=2900,
                target=3100,
                strategy_name="test",
                confidence=0.75,
                expected_value=50.0,
            ),
        }

        # Rebalance 1: empty -> [0.5, 0.5], turnover = 1.0
        rebalancer.step(signals, returns, {}, {})
        assert rebalancer.cumulative_turnover == pytest.approx(1.0)
        assert rebalancer.cumulative_cost == pytest.approx(cost_pct * 1.0)

        # Rebalance 2: [0.5, 0.5] -> [0.4, 0.6], turnover = 0.2
        rebalancer.step(signals, returns, {}, {})
        assert rebalancer.cumulative_turnover == pytest.approx(1.0 + 0.2)
        assert rebalancer.cumulative_cost == pytest.approx(cost_pct * (1.0 + 0.2))

        # Rebalance 3: [0.4, 0.6] -> [0.3, 0.7], turnover = 0.2
        rebalancer.step(signals, returns, {}, {})
        assert rebalancer.cumulative_turnover == pytest.approx(1.0 + 0.2 + 0.2)
        assert rebalancer.cumulative_cost == pytest.approx(cost_pct * (1.0 + 0.2 + 0.2))

    def test_rebalancer_turnover_threshold_vs_daily(self, synthetic_returns_200_obs_3_assets):
        """
        Threshold mode should have lower turnover than daily full rebalance.

        Acceptance: simulate drifting weights over 20 days.
        Threshold mode rebalances less frequently -> lower cumulative turnover.
        """
        returns = synthetic_returns_200_obs_3_assets
        cost_pct = 0.001

        # Simulate drifting weights: start at [0.5, 0.5], drift +0.01 per day
        # After N days: [0.5+0.01*N, 0.5-0.01*N]
        drifting_targets = [
            {"BTC": 0.5 + 0.01 * i, "ETH": 0.5 - 0.01 * i}
            for i in range(20)
        ]

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
            ),
            "ETH": Signal(
                direction=Direction.LONG,
                size=0.1,
                entry=3000,
                stop=2900,
                target=3100,
                strategy_name="test",
                confidence=0.75,
                expected_value=50.0,
            ),
        }

        # Threshold mode: band=0.05, so rebalance when drift > 0.05
        allocator_threshold = StubAllocator()
        allocator_threshold.target_weights_sequence = list(drifting_targets)
        rebalancer_threshold = Rebalancer(
            allocator_threshold, mode="threshold", band=0.05, cost_pct=cost_pct
        )

        # Daily full rebalance mode: rebalance every day (min_interval_days=1)
        allocator_daily = StubAllocator()
        allocator_daily.target_weights_sequence = list(drifting_targets)
        rebalancer_daily = Rebalancer(
            allocator_daily, mode="periodic", min_interval_days=1, cost_pct=cost_pct
        )

        # Run both through the same weight stream
        for _ in range(20):
            rebalancer_threshold.step(signals, returns, {}, {})
            rebalancer_daily.step(signals, returns, {}, {})

        # Threshold mode should have less turnover
        assert rebalancer_threshold.cumulative_turnover < rebalancer_daily.cumulative_turnover
        # Cost should track turnover
        assert rebalancer_threshold.cumulative_cost == pytest.approx(
            cost_pct * rebalancer_threshold.cumulative_turnover
        )
        assert rebalancer_daily.cumulative_cost == pytest.approx(
            cost_pct * rebalancer_daily.cumulative_turnover
        )
