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


# =============================================================================
# MVO ALLOCATOR
# =============================================================================

class MVOAllocator(PortfolioAllocator):
    """
    Mean-Variance Optimization allocator using Markowitz maximum-Sharpe portfolio.

    Uses Kronos-derived expected returns (from signal brackets) and shrunk covariance
    to solve the Sharpe ratio maximization problem:

        maximize (w·mu - rf) / sqrt(w'Σw)

    subject to:
        sum|w| <= gross_cap (leverage constraint)
        |w_i| <= max_weight (position constraint)
        sign(w_i) matches signal direction

    The allocator falls back to equal-weight allocation when fewer than min_obs
    (default 60) observations are available.

    Attributes:
        name: "mvo_allocator"
        lookback: Number of days of historical returns to use for covariance.
        gross_cap: Gross leverage cap (sum of |w_i|).
        max_weight: Maximum absolute weight for any single asset.
        rf: Risk-free rate (in same units as expected returns).
    """

    name: str = "mvo_allocator"

    def __init__(self, lookback: int = 120, gross_cap: float = 1.0,
                 max_weight: float = 0.35, rf: float = 0.0):
        """
        Initialize MVO allocator.

        Args:
            lookback: Number of days of historical returns to use for covariance estimation.
                     Default: 120 (approximately 6 months).
            gross_cap: Gross leverage cap (sum of absolute values of weights).
                      Default: 1.0 (dollar-neutral with 35% longs and 65% cash, etc.)
            max_weight: Maximum absolute weight for any single asset.
                       Default: 0.35 (35% maximum position size).
            rf: Risk-free rate (in same units as expected returns, typically daily).
               Default: 0.0.

        Examples:
            >>> allocator = MVOAllocator(lookback=120, gross_cap=1.0, max_weight=0.35, rf=0.0)
            >>> weights = allocator.allocate(signals, returns, dists, context)
        """
        self.lookback = lookback
        self.gross_cap = gross_cap
        self.max_weight = max_weight
        self.rf = rf

    def allocate(self, signals: Dict[str, "Signal"],
                 returns: pd.DataFrame,
                 dists: Dict[str, Any],
                 context: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute MVO-optimal weights.

        Args:
            signals: Dict[symbol -> Signal] of active signals (e.g., from generate_signal()).
            returns: pd.DataFrame of (n_obs, n_assets) daily log returns.
                    Index: dates, Columns: asset symbols.
            dists: Dict[symbol -> KairosDistribution] predicted distributions per asset.
            context: Dict[str, Any] execution context (unused here).

        Returns:
            Dict[symbol -> float] of target weights, signed (long/short).
            Sum of absolute values <= gross_cap.
            Individual absolute values <= max_weight.
            Signs match signal directions.

        Fall-back behavior:
            If len(returns) < self.min_obs (default 60), returns equal-weight allocation.
            If optimization fails for any reason, returns equal-weight allocation.

        Examples:
            >>> allocator = MVOAllocator()
            >>> weights = allocator.allocate(signals, returns, dists, {})
            >>> assert sum(abs(w) for w in weights.values()) <= 1.0
        """
        # Check if enough observations
        if len(returns) < self.min_obs:
            return _fallback_equal_weight(signals)

        # Extract symbols
        symbols = list(signals.keys())
        if not symbols:
            return {}

        # Get trailing returns for covariance estimation
        trailing_returns = returns[symbols].tail(self.lookback)
        if len(trailing_returns) < 2:
            return _fallback_equal_weight(signals)

        # Compute shrunk covariance
        try:
            cov = shrunk_covariance(trailing_returns)
        except Exception:
            return _fallback_equal_weight(signals)

        # Compute expected returns from Kronos distributions
        mu = []
        for sym in symbols:
            signal = signals[sym]
            dist = dists[sym]
            try:
                ev = dist.expected_value(
                    entry=signal.entry,
                    target=signal.target,
                    stop=signal.stop
                )
                mu.append(ev)
            except Exception:
                mu.append(0.0)
        mu = np.array(mu)

        # Solve MVO
        try:
            weights_dict = self._solve_mvo(symbols, mu, cov, signals)
            return weights_dict
        except Exception:
            # Fallback to equal weight on solver failure
            return _fallback_equal_weight(signals)

    def _solve_mvo(self, symbols: list, mu: np.ndarray, cov: np.ndarray,
                   signals: Dict[str, "Signal"]) -> Dict[str, float]:
        """
        Solve the Markowitz MVO problem via scipy.optimize.minimize with SLSQP.

        Maximizes (w·mu - rf) / sqrt(w'Σw) (Sharpe ratio) subject to:
        - sum|w| <= gross_cap
        - |w_i| <= max_weight
        - sign(w_i) matches signal direction

        Args:
            symbols: List of asset symbols (in order matching mu and cov).
            mu: Expected returns array (n_assets,).
            cov: Shrunk covariance matrix (n_assets, n_assets).
            signals: Dict[symbol -> Signal] for direction and metadata.

        Returns:
            Dict[symbol -> float] of optimal weights.

        Raises:
            ValueError: If optimization fails to converge.
        """
        from scipy.optimize import minimize

        # Import Direction locally to avoid circular dependency issues
        try:
            from kairos_backtest import Direction
        except ImportError:
            # Fallback if kairos_backtest not available
            class Direction:
                LONG = 1
                SHORT = -1
                FLAT = 0

        n = len(symbols)

        # Build bounds based on signal direction
        bounds = []
        for sym in symbols:
            direction = signals[sym].direction
            try:
                dir_val = direction.value if hasattr(direction, 'value') else direction
            except:
                dir_val = direction

            # Check if LONG (value=1), SHORT (value=-1), or FLAT (value=0)
            if dir_val == 1:  # LONG
                bounds.append((0.0, self.max_weight))
            elif dir_val == -1:  # SHORT
                bounds.append((-self.max_weight, 0.0))
            else:  # FLAT
                bounds.append((0.0, 0.0))

        # Objective: minimize negative Sharpe ratio
        def objective(w):
            # Portfolio return
            mu_port = np.dot(w, mu) - self.rf
            # Portfolio volatility
            vol_sq = np.dot(w, np.dot(cov, w))
            vol_port = np.sqrt(vol_sq + 1e-12)  # Add epsilon to avoid division issues

            if vol_port < 1e-10:
                return 1e10
            # Minimize negative Sharpe = maximize Sharpe
            return -mu_port / vol_port

        # Constraint: sum(|w|) <= gross_cap
        def constraint_gross_cap(w):
            return self.gross_cap - np.sum(np.abs(w))

        constraints = [
            {'type': 'ineq', 'fun': constraint_gross_cap}
        ]

        # Initial guess: equal weight
        w0 = np.ones(n) / n

        # Solve
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if not result.success or result.fun >= 1e9:
            raise ValueError(f"MVO optimization failed: {result.message}")

        # Return as dict
        weights = {}
        for i, sym in enumerate(symbols):
            w = result.x[i]
            # Zero out very small weights to avoid numerical noise
            if abs(w) < 1e-10:
                w = 0.0
            weights[sym] = float(w)

        return weights
