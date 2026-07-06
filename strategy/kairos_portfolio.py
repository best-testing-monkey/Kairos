"""
kairos_portfolio.py
===================
Portfolio allocation base class and covariance shrinkage for Kairos.

This module provides:
- PortfolioAllocator: abstract base class for position sizing across multiple assets
- shrunk_covariance: Ledoit-Wolf shrinkage toward scaled identity (sklearn optional)
- _fallback_equal_weight: equal-weight helper when observations < min_obs

The shrinkage function handles singular/near-singular covariance matrices by
shrinking toward a scaled identity matrix, maintaining positive definiteness even
when n_assets > n_obs (observations).

Usage
-----
    from kairos_portfolio import PortfolioAllocator, shrunk_covariance
    from kairos_backtest import Signal
    import pandas as pd
    import numpy as np

    # Subclass PortfolioAllocator to implement custom allocation logic
    class MyAllocator(PortfolioAllocator):
        def allocate(self, signals, returns, dists, context):
            # signals: Dict[str, Signal]
            # returns: pd.DataFrame (n_obs x n_assets)
            # dists: Dict[str, KairosDistribution]
            # context: Dict[str, Any]
            cov = shrunk_covariance(returns)
            # ... solve QP / MVO / etc. using cov ...
            return weights  # Dict[symbol -> float]

    allocator = MyAllocator()
    weights = allocator.allocate(signals, returns, dists, {})
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any

try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# SHRINKAGE COVARIANCE
# =============================================================================

def shrunk_covariance(returns: pd.DataFrame, shrinkage_intensity: Optional[float] = None) -> np.ndarray:
    """
    Compute a shrunk covariance matrix via Ledoit-Wolf toward scaled identity.

    The Ledoit-Wolf shrinkage estimator reduces estimation error when the sample
    covariance is singular or ill-conditioned (e.g., when n_assets > n_obs).
    The target is a scaled identity matrix: T = tr(S)/k * I.

    Args:
        returns: DataFrame of shape (n_obs, n_assets) with daily log returns.
                 NaN/inf rows are skipped.
        shrinkage_intensity: Optional manual shrinkage α ∈ [0,1].
                            If provided, used directly (for testing).
                            Otherwise, computed via Ledoit-Wolf closed-form.

    Returns:
        Shrunk covariance matrix of shape (n_assets, n_assets).
        Always positive definite (eigenvalues >= tr(S)/(k*(n_obs+1)) * min_eigenval_contribution).

    References:
        Ledoit, O., & Wolf, M. (2004).
        "Honey, I shrunk the sample covariance matrix."
        The Journal of Portfolio Management, 30(4), 110-119.
    """
    # Clean data: drop NaN/inf rows
    ret = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    if len(ret) < 2:
        # Fallback: zero matrix if insufficient data
        n_assets = returns.shape[1]
        return np.eye(n_assets)

    n_obs, n_assets = ret.shape

    # Compute sample covariance
    S = np.cov(ret.T)

    # Handle degenerate case (single asset)
    if S.ndim == 0:
        S = np.array([[float(S)]])

    # Compute shrinkage intensity if not provided
    if shrinkage_intensity is None:
        if HAS_SKLEARN:
            # Use sklearn's LedoitWolf
            lw = LedoitWolf()
            _, alpha = lw.fit(ret).covariance_, lw.shrinkage_
        else:
            # Manual Ledoit-Wolf closed-form shrinkage
            alpha = _ledoit_wolf_intensity(ret, S)
    else:
        alpha = float(np.clip(shrinkage_intensity, 0.0, 1.0))

    # Target: scaled identity
    trace_S = np.trace(S)
    target = (trace_S / n_assets) * np.eye(n_assets)

    # Shrink: Σ_shrunk = (1-α)*S + α*T
    cov_shrunk = (1.0 - alpha) * S + alpha * target

    # Ensure positive definiteness via eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(cov_shrunk)
    # Clip negative/zero eigenvalues to a small positive value
    min_eigval = np.abs(trace_S) / (n_assets * (n_obs + 1))
    eigvals = np.maximum(eigvals, min_eigval)
    cov_shrunk = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return cov_shrunk


def _ledoit_wolf_intensity(returns: pd.DataFrame, sample_cov: np.ndarray) -> float:
    """
    Compute Ledoit-Wolf shrinkage intensity via closed-form formula.

    Shrinks sample covariance toward scaled identity matrix.

    Args:
        returns: DataFrame of shape (n_obs, n_assets), cleaned.
        sample_cov: Sample covariance matrix S.

    Returns:
        Shrinkage intensity α ∈ [0,1].
    """
    n_obs, n_assets = returns.shape

    # Shrink toward target T = (tr(S)/k) * I
    trace_S = np.trace(sample_cov)
    target = (trace_S / n_assets) * np.eye(n_assets)

    # Ledoit-Wolf closed-form: α = ((1 - 2/k)*tr(S²) + tr(S)²) / ((n+1-2/k)*(tr(S²) - tr(S)²/k))
    S2 = sample_cov @ sample_cov
    tr_S2 = np.trace(S2)
    tr_S = trace_S

    numerator = (1.0 - 2.0 / n_assets) * tr_S2 + tr_S ** 2
    denominator = (n_obs + 1.0 - 2.0 / n_assets) * (tr_S2 - tr_S ** 2 / n_assets)

    if np.abs(denominator) < 1e-10:
        # Avoid division by zero
        return 0.5

    alpha = numerator / denominator
    alpha = float(np.clip(alpha, 0.0, 1.0))

    return alpha


# =============================================================================
# PORTFOLIO ALLOCATOR BASE CLASS
# =============================================================================

class PortfolioAllocator:
    """
    Abstract base class for portfolio allocation.

    An allocator consumes per-asset signals and returns target weights
    (signed, summing to at most 1.0 in absolute value).

    Attributes:
        name: String identifier for the allocator (e.g., "mvo_allocator").
        min_obs: Minimum observations required to solve the allocation problem.
                 Below this threshold, falls back to equal weight.
    """

    name: str = "base_allocator"
    min_obs: int = 60

    def allocate(
        self,
        signals: Dict[str, "Signal"],
        returns: pd.DataFrame,
        dists: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute target portfolio weights from signals and returns.

        Args:
            signals: Dict[symbol -> Signal], where Signal is from kairos_backtest.
                    May be empty (no trades) or contain signals for a subset of assets.
            returns: pd.DataFrame of shape (n_obs, n_assets) with daily log returns.
                    Index: dates, Columns: asset symbols.
                    May have n_obs < self.min_obs, triggering fallback.
            dists: Dict[symbol -> KairosDistribution], predicted distributions.
            context: Dict[str, Any], execution context (may include realized_vol, etc.).

        Returns:
            Dict[symbol -> float] target weights. Weights should be signed
            (long/short) and sum of absolute values should not exceed 1.0
            (gross leverage cap).

        Raises:
            NotImplementedError: Subclasses must override this method.

        Examples:
            >>> allocator = MyAllocator()
            >>> weights = allocator.allocate(signals, returns, dists, {})
            >>> assert isinstance(weights, dict)
            >>> assert all(isinstance(v, float) for v in weights.values())
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.allocate() must be implemented by subclasses"
        )

    def reset(self) -> None:
        """
        Reset any internal state (for stateful allocators).

        Called at the start of walk-forward folds to prevent state leakage.
        Default implementation is a no-op; override in stateful subclasses
        (e.g., universal portfolio, GA allocator).
        """
        pass


def _fallback_equal_weight(signals: Dict[str, "Signal"]) -> Dict[str, float]:
    """
    Equal-weight allocation helper for when observations < min_obs.

    Distributes weight equally among all signaled assets. Used when
    insufficient history prevents solving the primary allocation problem.

    Args:
        signals: Dict[symbol -> Signal], typically non-empty.

    Returns:
        Dict[symbol -> weight], where each weight = 1 / len(signals)
        if len(signals) > 0, else empty dict.

    Examples:
        >>> signals = {"BTC": Signal(...), "ETH": Signal(...)}
        >>> weights = _fallback_equal_weight(signals)
        >>> weights
        {'BTC': 0.5, 'ETH': 0.5}
        >>> sum(weights.values())
        1.0
    """
    if not signals:
        return {}

    n = len(signals)
    weight = 1.0 / n
    return {symbol: weight for symbol in signals.keys()}
