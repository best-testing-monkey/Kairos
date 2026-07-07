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
from typing import Dict, Optional, Any, List, Tuple

try:
    from sklearn.covariance import LedoitWolf
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


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
# RISK PARITY ALLOCATOR
# =============================================================================

class RiskParityAllocator(PortfolioAllocator):
    """
    Equal Risk Contribution (ERC) allocator using the Spinu (2013) formulation.

    Solves for weights w where each asset contributes equally to portfolio risk:
    w_i * (Σw)_i equal for all i.

    Optimization (convex, over positive magnitudes y):
        minimize: 0.5 * y'Σy - c * sum(log(y_i))

    The first-order condition y_i (Σy)_i = c yields exactly equal risk
    contributions at the optimum. Magnitudes are then normalized to sum to 1,
    uniformly scaled to respect max_weight/gross_cap (uniform scaling preserves
    ERC), and signed per each asset's signal direction (LONG/SHORT/FLAT=0).

    Risk contribution of asset i is:
        rc_i = w_i * (Σw)_i

    Attributes:
        name: "risk_parity_allocator"
        lookback: Number of days of historical returns for covariance estimation.
        gross_cap: Gross leverage cap (sum of |w_i|).
        max_weight: Maximum absolute weight for any single asset.
    """

    name: str = "risk_parity_allocator"

    def __init__(self, lookback: int = 120, gross_cap: float = 1.0,
                 max_weight: float = 0.35):
        """
        Initialize Risk Parity allocator.

        Args:
            lookback: Number of days of historical returns to use for covariance estimation.
                     Default: 120 (approximately 6 months).
            gross_cap: Gross leverage cap (sum of absolute values of weights).
                      Default: 1.0.
            max_weight: Maximum absolute weight for any single asset.
                       Default: 0.35.

        Examples:
            >>> allocator = RiskParityAllocator(lookback=120, gross_cap=1.0, max_weight=0.35)
            >>> weights = allocator.allocate(signals, returns, dists, context)
        """
        self.lookback = lookback
        self.gross_cap = gross_cap
        self.max_weight = max_weight

    def allocate(self, signals: Dict[str, "Signal"],
                 returns: pd.DataFrame,
                 dists: Dict[str, Any],
                 context: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute ERC-optimal weights.

        Args:
            signals: Dict[symbol -> Signal] of active signals.
            returns: pd.DataFrame of (n_obs, n_assets) daily log returns.
                    Index: dates, Columns: asset symbols.
            dists: Dict[symbol -> KairosDistribution] (unused for ERC).
            context: Dict[str, Any] execution context (unused).

        Returns:
            Dict[symbol -> float] of target weights, signed (long/short).
            Sum of absolute values <= gross_cap.
            Individual absolute values <= max_weight.
            Signs match signal directions.

        Fall-back behavior:
            If len(returns) < self.min_obs (default 60), returns equal-weight allocation.
            If optimization fails, returns equal-weight allocation.

        Examples:
            >>> allocator = RiskParityAllocator()
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

        # Solve ERC
        try:
            weights_dict = self._solve_erc(symbols, cov, signals)
            return weights_dict
        except Exception:
            # Fallback to equal weight on solver failure
            return _fallback_equal_weight(signals)

    def _solve_erc(self, symbols: list, cov: np.ndarray,
                   signals: Dict[str, "Signal"]) -> Dict[str, float]:
        """
        Solve the Equal Risk Contribution problem via the Spinu (2013) formulation.

        Optimizes over unnormalized positive magnitudes y > 0 (active assets only):

            minimize  0.5 * y'Σ*y - c * sum(log(y_i))

        where Σ* is the sign-adjusted covariance (Σ*_ij = s_i s_j Σ_ij with s the
        signal-direction signs). This problem is strictly convex; its first-order
        condition (Σ*y)_i = c / y_i implies y_i (Σ*y)_i = c for all i, i.e. exactly
        equal risk contributions at the optimum. The signed weights w = s ∘ y then
        satisfy w_i (Σw)_i = c as well.

        Post-processing:
        - Normalize |w| to sum to 1 (magnitudes from ERC, per ticket).
        - Scale down uniformly if any |w_i| > max_weight or sum|w| > gross_cap
          (uniform scaling preserves the ERC property).
        - Apply signal-direction signs.

        Args:
            symbols: List of asset symbols (in order matching cov).
            cov: Shrunk covariance matrix (n_assets, n_assets).
            signals: Dict[symbol -> Signal] for direction and metadata.

        Returns:
            Dict[symbol -> float] of optimal weights.

        Raises:
            ValueError: If optimization fails to converge.

        References:
            Spinu, F. (2013). "An Algorithm for Computing Risk Parity Weights."
        """
        from scipy.optimize import minimize

        n = len(symbols)

        # Extract signal-direction signs; FLAT assets are excluded from the solve
        signs = np.zeros(n)
        for i, sym in enumerate(symbols):
            direction = signals[sym].direction
            try:
                dir_val = direction.value if hasattr(direction, 'value') else direction
            except:
                dir_val = direction
            if dir_val == 1:
                signs[i] = 1.0
            elif dir_val == -1:
                signs[i] = -1.0
            # FLAT stays 0

        active = np.where(signs != 0.0)[0]
        if len(active) == 0:
            return {sym: 0.0 for sym in symbols}

        # Sign-adjusted covariance restricted to active assets:
        # Σ*_ij = s_i s_j Σ_ij, so ERC on y>0 with Σ* equals ERC on signed w with Σ.
        s_act = signs[active]
        cov_act = cov[np.ix_(active, active)] * np.outer(s_act, s_act)

        # Rescale covariance to O(1) to avoid numerical underflow in the solver
        # (daily-return covariances are ~1e-4). Uniform scaling does not change
        # the ERC solution direction.
        scale = np.mean(np.diag(cov_act))
        if scale <= 0:
            raise ValueError("Non-positive covariance diagonal")
        cov_s = cov_act / scale

        k = len(active)
        c = 1.0 / k  # log-barrier weight; any c > 0 gives the same normalized weights

        def objective(y):
            return 0.5 * np.dot(y, cov_s @ y) - c * np.sum(np.log(y))

        def gradient(y):
            return cov_s @ y - c / y

        # Initial guess: inverse-volatility magnitudes (near the ERC optimum)
        vols = np.sqrt(np.diag(cov_s))
        y0 = (1.0 / vols)
        y0 = y0 / np.sum(y0)

        result = minimize(
            objective,
            y0,
            jac=gradient,
            method='SLSQP',
            bounds=[(1e-8, None)] * k,
            options={'ftol': 1e-12, 'maxiter': 1000}
        )

        if not result.success:
            raise ValueError(f"ERC optimization failed: {result.message}")

        y = result.x

        # Normalize magnitudes to sum to 1
        total = np.sum(y)
        if total <= 0:
            raise ValueError("Degenerate ERC solution")
        mag = y / total

        # Uniformly scale down to respect max_weight and gross_cap
        # (uniform scaling preserves relative risk contributions)
        shrink = 1.0
        max_mag = np.max(mag)
        if max_mag > self.max_weight:
            shrink = min(shrink, self.max_weight / max_mag)
        if np.sum(mag) > self.gross_cap:
            shrink = min(shrink, self.gross_cap / np.sum(mag))
        mag = mag * shrink

        # Assemble signed weights
        weights = {sym: 0.0 for sym in symbols}
        for j, idx in enumerate(active):
            w = float(signs[idx] * mag[j])
            if abs(w) < 1e-10:
                w = 0.0
            weights[symbols[idx]] = w

        return weights


# =============================================================================
# HIERARCHICAL RISK PARITY ALLOCATOR
# =============================================================================

class HRPAllocator(PortfolioAllocator):
    """
    Hierarchical Risk Parity (HRP) allocator using correlation-distance clustering.

    López de Prado's HRP algorithm:
    1. Correlation matrix → distance matrix: dist_ij = sqrt(0.5 * (1 - corr_ij))
    2. Hierarchical clustering: scipy.cluster.hierarchy.linkage(method="single")
    3. Quasi-diagonalization: extract leaf order (seriation) from dendrogram
    4. Recursive bisection: split clusters and allocate inversely to variance
    5. No matrix inversion required; robust when n_assets approaches n_obs

    Attributes:
        name: "hrp_allocator"
        lookback: Number of days of historical returns for covariance estimation.
        variant: "hrp" for inverse-variance allocation, "herc" for equal risk contribution.
    """

    name: str = "hrp_allocator"

    def __init__(self, lookback: int = 120, variant: str = "hrp"):
        """
        Initialize HRP allocator.

        Args:
            lookback: Number of days of historical returns to use for covariance estimation.
                     Default: 120 (approximately 6 months).
            variant: "hrp" for inverse-variance splits (default), or "herc" for
                    hierarchical equal risk contribution (equal risk per cluster).

        Examples:
            >>> allocator = HRPAllocator(lookback=120, variant="hrp")
            >>> weights = allocator.allocate(signals, returns, dists, context)
        """
        if variant not in ("hrp", "herc"):
            raise ValueError(f"variant must be 'hrp' or 'herc', got '{variant}'")
        self.lookback = lookback
        self.variant = variant

    def allocate(self, signals: Dict[str, "Signal"],
                 returns: pd.DataFrame,
                 dists: Dict[str, Any],
                 context: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute HRP-optimal weights via correlation-distance clustering and recursive bisection.

        Args:
            signals: Dict[symbol -> Signal] of active signals.
            returns: pd.DataFrame of (n_obs, n_assets) daily log returns.
                    Index: dates, Columns: asset symbols.
            dists: Dict[symbol -> KairosDistribution] (unused for HRP).
            context: Dict[str, Any] execution context (unused).

        Returns:
            Dict[symbol -> float] of target weights, signed (long/short).
            Sum of absolute values typically close to 1.0 (subject to signal directions).
            Magnitudes follow HRP allocation; signs follow signal directions.

        Fall-back behavior:
            If len(returns) < self.min_obs (default 60), returns equal-weight allocation.
            If n_assets < 2, raises ValueError (HRP requires at least 2 assets).
            If clustering fails, returns equal-weight allocation.

        Examples:
            >>> allocator = HRPAllocator()
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

        # Special case: n_assets == 1 (no clustering possible)
        if len(symbols) == 1:
            # Allocate all weight to single asset
            sym = symbols[0]
            direction = signals[sym].direction

            # Extract direction value (handles both enum and raw int)
            if hasattr(direction, 'value'):
                dir_val = direction.value
            else:
                dir_val = direction

            if dir_val == 1:
                return {sym: 1.0}
            elif dir_val == -1:
                return {sym: -1.0}
            else:
                return {sym: 0.0}

        # Get trailing returns for covariance estimation
        trailing_returns = returns[symbols].tail(self.lookback)
        if len(trailing_returns) < 2:
            return _fallback_equal_weight(signals)

        # Compute shrunk covariance
        try:
            cov = shrunk_covariance(trailing_returns)
        except Exception:
            return _fallback_equal_weight(signals)

        # Solve HRP
        try:
            # Special case: n_assets == 2 → degenerate to inverse-variance
            if len(symbols) == 2:
                weights_dict = self._solve_hrp_2asset(symbols, cov, signals)
            else:
                weights_dict = self._solve_hrp(symbols, cov, signals, trailing_returns)
            return weights_dict
        except Exception:
            # Fallback to equal weight on solver failure
            return _fallback_equal_weight(signals)

    def _solve_hrp_2asset(self, symbols: List[str], cov: np.ndarray,
                         signals: Dict[str, "Signal"]) -> Dict[str, float]:
        """
        Special case: n_assets=2 degenerates to inverse-variance allocation.

        Args:
            symbols: List of 2 asset symbols.
            cov: Covariance matrix (2x2).
            signals: Dict[symbol -> Signal] for direction and metadata.

        Returns:
            Dict[symbol -> float] with weights inversely proportional to variance.
        """
        assert len(symbols) == 2, "This function is only for 2-asset case"

        # Extract variances
        var0 = cov[0, 0]
        var1 = cov[1, 1]

        if var0 <= 0 or var1 <= 0:
            # Degenerate case: use equal weight
            return _fallback_equal_weight(signals)

        # Inverse-variance weights (magnitudes)
        inv_vol0 = 1.0 / np.sqrt(var0)
        inv_vol1 = 1.0 / np.sqrt(var1)
        total_inv_vol = inv_vol0 + inv_vol1

        mag0 = inv_vol0 / total_inv_vol
        mag1 = inv_vol1 / total_inv_vol

        # Apply signal directions
        weights = {}
        for i, sym in enumerate(symbols):
            mag = mag0 if i == 0 else mag1
            direction = signals[sym].direction

            # Extract direction value (handles both enum and raw int)
            if hasattr(direction, 'value'):
                dir_val = direction.value
            else:
                dir_val = direction

            if dir_val == 1:  # LONG
                w = float(mag)
            elif dir_val == -1:  # SHORT
                w = float(-mag)
            else:  # FLAT or other
                w = 0.0

            if abs(w) < 1e-10:
                w = 0.0
            weights[sym] = w

        return weights

    def _solve_hrp(self, symbols: List[str], cov: np.ndarray,
                  signals: Dict[str, "Signal"],
                  trailing_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Solve HRP via correlation-distance clustering and recursive bisection.

        Args:
            symbols: List of asset symbols (in order matching cov).
            cov: Shrunk covariance matrix (n_assets, n_assets).
            signals: Dict[symbol -> Signal] for direction and metadata.
            trailing_returns: DataFrame of trailing returns (for computing clustering).

        Returns:
            Dict[symbol -> float] of HRP-optimal weights.

        Raises:
            ValueError: If clustering or allocation fails.
        """
        n = len(symbols)

        # Step 1: Compute correlation matrix
        corr = np.corrcoef(trailing_returns.T)

        # Handle NaN in correlation (e.g., constant series)
        if np.isnan(corr).any():
            # Fallback to equal weight
            raise ValueError("Correlation matrix contains NaN")

        # Step 2: Convert to distance matrix
        # dist_ij = sqrt(0.5 * (1 - corr_ij)), clipped to [0, 2] for numerical stability
        dist = np.sqrt(0.5 * (1.0 - corr))
        dist = np.clip(dist, 0.0, 2.0)
        # Ensure symmetry and zero diagonal (for numerical stability with squareform)
        dist = (dist + dist.T) / 2.0  # Make symmetric
        np.fill_diagonal(dist, 0.0)   # Zero diagonal

        # Step 3: Convert to condensed distance for linkage
        # squareform converts from square matrix to condensed form
        condensed_dist = squareform(dist)

        # Step 4: Hierarchical clustering (single linkage)
        Z = linkage(condensed_dist, method="single")

        # Step 5: Quasi-diagonalize (extract leaf order from dendrogram)
        # dendrogram returns a dict with the 'leaves' key giving the reordered indices
        dendro = dendrogram(Z, no_plot=True)
        leaf_order = dendro["leaves"]

        # Step 6: Recursive bisection with inverse-variance allocation
        # Start with all assets
        cluster = sorted(leaf_order)  # List of indices in the cluster

        # Compute magnitudes via recursive bisection
        magnitudes = np.zeros(n)
        self._recursive_bisection(cluster, magnitudes, cov, leaf_order)

        # Normalize magnitudes to sum to 1
        total = np.sum(np.abs(magnitudes))
        if total > 0:
            magnitudes = magnitudes / total
        else:
            # Fallback to equal weight
            magnitudes = np.ones(n) / n

        # Apply signal directions
        weights = {}
        for i, sym in enumerate(symbols):
            mag = magnitudes[i]
            direction = signals[sym].direction

            # Extract direction value (handles both enum and raw int)
            if hasattr(direction, 'value'):
                dir_val = direction.value
            else:
                dir_val = direction

            if dir_val == 1:  # LONG
                w = float(mag)
            elif dir_val == -1:  # SHORT
                w = float(-mag)
            else:  # FLAT or other
                w = 0.0

            if abs(w) < 1e-10:
                w = 0.0
            weights[sym] = w

        return weights

    def _recursive_bisection(self, cluster: List[int], magnitudes: np.ndarray,
                            cov: np.ndarray, leaf_order: List[int],
                            parent_mag: float = 1.0) -> None:
        """
        Recursive bisection: split cluster and allocate weights inversely to variance.

        Args:
            cluster: List of asset indices in current cluster.
            magnitudes: Output array to store magnitude for each asset (in place).
            cov: Full covariance matrix.
            leaf_order: Leaf order from dendrogram (for quasi-diagonalization).
            parent_mag: Total magnitude to allocate to this cluster (default 1.0).
        """
        if len(cluster) == 1:
            # Leaf node: assign parent_mag to this asset
            magnitudes[cluster[0]] = parent_mag
            return

        # Split cluster into two halves (using leaf_order for continuity)
        # Find position of each cluster member in leaf_order
        positions = [leaf_order.index(idx) for idx in cluster]
        positions_sorted = sorted(zip(positions, cluster))  # Sort by position in leaf_order

        # Split at midpoint
        split_idx = len(positions_sorted) // 2
        left_cluster = [idx for _, idx in positions_sorted[:split_idx]]
        right_cluster = [idx for _, idx in positions_sorted[split_idx:]]

        if len(left_cluster) == 0 or len(right_cluster) == 0:
            # Degenerate split: assign equal weight to all
            for idx in cluster:
                magnitudes[idx] = parent_mag / len(cluster)
            return

        # Compute variance of each cluster
        # Cluster variance = variance of equal-weight portfolio within cluster
        # = (1/n^2) * 1' * Sigma_cluster * 1, where n = cluster size
        # = (1/n^2) * sum of all elements in Sigma_cluster
        cov_left = cov[np.ix_(left_cluster, left_cluster)]
        cov_right = cov[np.ix_(right_cluster, right_cluster)]

        n_left = len(left_cluster)
        n_right = len(right_cluster)

        # Variance of equal-weight portfolio
        var_left = np.sum(cov_left) / (n_left ** 2)
        var_right = np.sum(cov_right) / (n_right ** 2)

        if var_left <= 0 or var_right <= 0:
            # Degenerate case
            for idx in cluster:
                magnitudes[idx] = parent_mag / len(cluster)
            return

        if self.variant == "herc":
            # HERC: equal risk contribution between clusters
            # Allocate equally between clusters, then recursively bisect within each
            left_mag = parent_mag * 0.5
            right_mag = parent_mag * 0.5
        else:
            # HRP (default): inverse-variance allocation
            inv_var_left = 1.0 / var_left
            inv_var_right = 1.0 / var_right
            total_inv_var = inv_var_left + inv_var_right

            left_mag = parent_mag * inv_var_left / total_inv_var
            right_mag = parent_mag * inv_var_right / total_inv_var

        # Recursively bisect left and right clusters
        self._recursive_bisection(left_cluster, magnitudes, cov, leaf_order, left_mag)
        self._recursive_bisection(right_cluster, magnitudes, cov, leaf_order, right_mag)


# =============================================================================
# MVO ALLOCATOR
# =============================================================================

class MinVarAllocator(PortfolioAllocator):
    """
    Minimum-Variance portfolio allocator with Ledoit-Wolf shrinkage.

    Minimizes portfolio variance w'Σw where Σ is a shrunk covariance matrix
    (Ledoit-Wolf toward scaled identity) over the signaled assets' trailing returns.
    Uses scipy SLSQP solver.

    Constraints:
        sum|w| <= gross_cap (gross leverage cap, default 1.0)
        |w_i| <= max_weight (individual position cap, default 0.35)
        sign(w_i) matches signal direction (LONG/SHORT/FLAT)

    The allocator falls back to equal-weight allocation when fewer than min_obs
    (default 60) observations are available, or if the optimizer fails to converge.

    Attributes:
        name: "minvar_allocator"
        lookback: Number of days of historical returns to use for covariance estimation.
        gross_cap: Gross leverage cap (sum of |w_i|).
        max_weight: Maximum absolute weight for any single asset.
    """

    name: str = "minvar_allocator"

    def __init__(self, lookback: int = 120, gross_cap: float = 1.0,
                 max_weight: float = 0.35):
        """
        Initialize MinVar allocator.

        Args:
            lookback: Number of days of historical returns to use for covariance estimation.
                     Default: 120 (approximately 6 months).
            gross_cap: Gross leverage cap (sum of absolute values of weights).
                      Default: 1.0.
            max_weight: Maximum absolute weight for any single asset.
                       Default: 0.35.

        Examples:
            >>> allocator = MinVarAllocator(lookback=120, gross_cap=1.0, max_weight=0.35)
            >>> weights = allocator.allocate(signals, returns, dists, context)
        """
        self.lookback = lookback
        self.gross_cap = gross_cap
        self.max_weight = max_weight

    def allocate(self, signals: Dict[str, "Signal"],
                 returns: pd.DataFrame,
                 dists: Dict[str, Any],
                 context: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute minimum-variance optimal weights.

        Args:
            signals: Dict[symbol -> Signal] of active signals.
            returns: pd.DataFrame of (n_obs, n_assets) daily log returns.
                    Index: dates, Columns: asset symbols.
            dists: Dict[symbol -> KairosDistribution] (unused for MinVar).
            context: Dict[str, Any] execution context (unused).

        Returns:
            Dict[symbol -> float] of target weights, signed (long/short).
            Sum of absolute values <= gross_cap.
            Individual absolute values <= max_weight.
            Signs match signal directions.

        Fall-back behavior:
            If len(returns) < self.min_obs (default 60), returns equal-weight allocation.
            If optimization fails, returns equal-weight allocation.

        Examples:
            >>> allocator = MinVarAllocator()
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

        # Solve MinVar
        try:
            weights_dict = self._solve_minvar(symbols, cov, signals)
            return weights_dict
        except Exception:
            # Fallback to equal weight on solver failure
            return _fallback_equal_weight(signals)

    def _solve_minvar(self, symbols: list, cov: np.ndarray,
                      signals: Dict[str, "Signal"]) -> Dict[str, float]:
        """
        Solve the minimum-variance problem via scipy.optimize.minimize with SLSQP.

        Minimizes w'Σw subject to:
        - sum|w| <= gross_cap
        - |w_i| <= max_weight
        - sign(w_i) matches signal direction

        The objective includes a small penalty on under-allocation to avoid the
        trivial zero-weight solution while preserving variance minimization.

        Args:
            symbols: List of asset symbols (in order matching cov).
            cov: Shrunk covariance matrix (n_assets, n_assets).
            signals: Dict[symbol -> Signal] for direction and metadata.

        Returns:
            Dict[symbol -> float] of optimal weights.

        Raises:
            ValueError: If optimization fails to converge.
        """
        from scipy.optimize import minimize

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

        # Objective: minimize portfolio variance w'Σw with penalty for under-allocation
        # penalty discourages trivial zero-weight solution while preserving variance minimization
        def objective(w):
            variance = np.dot(w, np.dot(cov, w))
            # Compute maximum feasible allocation (constrained by bounds)
            max_alloc = sum(abs(ub) if ub * lb <= 0 else max(abs(ub), abs(lb))
                           for lb, ub in bounds)
            # Feasible target: min(gross_cap, max_alloc)
            target_alloc = min(self.gross_cap, max_alloc)
            # Penalize under-allocation relative to feasible target
            # Use modest penalty to encourage meaningful allocation without dominating variance
            under_alloc = target_alloc - np.sum(np.abs(w))
            penalty = 1e-4 * max(0, under_alloc) ** 2  if target_alloc > 1e-10 else 0.0
            return variance + penalty

        # Gradient approximation via finite differences (SLSQP can compute it)
        # (letting SLSQP use numerical gradient is simpler given the penalty term)

        # Constraint: sum(|w|) <= gross_cap (inequality)
        def constraint_gross_cap(w):
            return self.gross_cap - np.sum(np.abs(w))

        constraints = [
            {'type': 'ineq', 'fun': constraint_gross_cap}
        ]

        # Initial guess: equal weight scaled to gross_cap, clipped to bounds
        w0 = np.ones(n) / n * self.gross_cap
        # Clip to bounds
        for i, (lb, ub) in enumerate(bounds):
            w0[i] = np.clip(w0[i], lb, ub)

        # Solve
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if not result.success:
            raise ValueError(f"MinVar optimization failed: {result.message}")

        # Return as dict
        weights = {}
        for i, sym in enumerate(symbols):
            w = result.x[i]
            # Zero out very small weights to avoid numerical noise
            if abs(w) < 1e-10:
                w = 0.0
            weights[sym] = float(w)

        return weights


# =============================================================================
# MAX-SHARPE SOLVER (shared by MVO and Black-Litterman)
# =============================================================================

def _max_sharpe_solve(
    symbols: list,
    mu: np.ndarray,
    cov: np.ndarray,
    signals: Dict[str, "Signal"],
    gross_cap: float = 1.0,
    max_weight: float = 0.35,
    rf: float = 0.0,
) -> Dict[str, float]:
    """
    Solve the Markowitz max-Sharpe problem via scipy.optimize.minimize with SLSQP.

    Maximizes (w·mu - rf) / sqrt(w'Σw) (Sharpe ratio) subject to:
    - sum|w| <= gross_cap
    - |w_i| <= max_weight
    - sign(w_i) matches signal direction

    This is a module-level helper used by both MVOAllocator and BlackLittermanAllocator.

    Args:
        symbols: List of asset symbols (in order matching mu and cov).
        mu: Expected returns array (n_assets,).
        cov: Shrunk covariance matrix (n_assets, n_assets).
        signals: Dict[symbol -> Signal] for direction and metadata.
        gross_cap: Gross leverage cap (sum of |w_i|). Default: 1.0.
        max_weight: Maximum absolute weight for any single asset. Default: 0.35.
        rf: Risk-free rate (in same units as expected returns). Default: 0.0.

    Returns:
        Dict[symbol -> float] of optimal weights.

    Raises:
        ValueError: If optimization fails to converge.
    """
    from scipy.optimize import minimize

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
            bounds.append((0.0, max_weight))
        elif dir_val == -1:  # SHORT
            bounds.append((-max_weight, 0.0))
        else:  # FLAT
            bounds.append((0.0, 0.0))

    # Objective: minimize negative Sharpe ratio
    def objective(w):
        # Portfolio return
        mu_port = np.dot(w, mu) - rf
        # Portfolio volatility
        vol_sq = np.dot(w, np.dot(cov, w))
        vol_port = np.sqrt(vol_sq + 1e-12)  # Add epsilon to avoid division issues

        if vol_port < 1e-10:
            return 1e10
        # Minimize negative Sharpe = maximize Sharpe
        return -mu_port / vol_port

    # Constraint: sum(|w|) <= gross_cap
    def constraint_gross_cap(w):
        return gross_cap - np.sum(np.abs(w))

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
        raise ValueError(f"Max-Sharpe optimization failed: {result.message}")

    # Return as dict
    weights = {}
    for i, sym in enumerate(symbols):
        w = result.x[i]
        # Zero out very small weights to avoid numerical noise
        if abs(w) < 1e-10:
            w = 0.0
        weights[sym] = float(w)

    return weights


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

        # Solve MVO using shared solver
        try:
            weights_dict = _max_sharpe_solve(
                symbols, mu, cov, signals,
                gross_cap=self.gross_cap,
                max_weight=self.max_weight,
                rf=self.rf
            )
            return weights_dict
        except Exception:
            # Fallback to equal weight on solver failure
            return _fallback_equal_weight(signals)


# =============================================================================
# BLACK-LITTERMAN POSTERIOR HELPER
# =============================================================================

def _bl_posterior(
    pi: np.ndarray,
    Q: np.ndarray,
    Sigma: np.ndarray,
    Omega: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """
    Compute Black-Litterman posterior expected returns.

    Given:
    - Prior equilibrium returns Π (n_assets,)
    - Views Q (n_assets,) — absolute return expectations per asset
    - Covariance matrix Σ (n_assets, n_assets)
    - View uncertainty Ω (n_assets, n_assets) — diagonal matrix
    - Confidence parameter τ (scalar)

    Returns:
    - Posterior mu_BL = inv(inv(τΣ) + P'inv(Ω)P) @ (inv(τΣ)Π + P'inv(Ω)Q)
      where P = I (one absolute view per asset)

    Args:
        pi: Prior equilibrium returns (n_assets,).
        Q: Views on returns (n_assets,).
        Sigma: Covariance matrix (n_assets, n_assets).
        Omega: View uncertainty (n_assets, n_assets), typically diagonal.
        tau: Confidence scaling parameter. Default: 0.05.

    Returns:
        Posterior mean returns (n_assets,).

    Raises:
        ValueError: If matrix inversion fails.
    """
    n = len(pi)

    # Compute inv(τΣ)
    tau_sigma = tau * Sigma
    try:
        inv_tau_sigma = np.linalg.inv(tau_sigma)
    except np.linalg.LinAlgError:
        raise ValueError("Failed to invert τΣ")

    # Compute inv(Ω)
    try:
        inv_omega = np.linalg.inv(Omega)
    except np.linalg.LinAlgError:
        raise ValueError("Failed to invert Ω")

    # With P = I (n x n identity), P'inv(Ω)P = inv(Ω)
    # Posterior precision: inv(τΣ) + inv(Ω)
    posterior_precision = inv_tau_sigma + inv_omega

    # Posterior mean: inv(posterior_precision) @ (inv(τΣ)Π + inv(Ω)Q)
    try:
        inv_posterior_precision = np.linalg.inv(posterior_precision)
    except np.linalg.LinAlgError:
        raise ValueError("Failed to invert posterior precision matrix")

    rhs = inv_tau_sigma @ pi + inv_omega @ Q
    mu_bl = inv_posterior_precision @ rhs

    return mu_bl


# =============================================================================
# BLACK-LITTERMAN ALLOCATOR
# =============================================================================

class BlackLittermanAllocator(PortfolioAllocator):
    """
    Black-Litterman allocator combining equilibrium prior with Kronos-derived views.

    Prior: equilibrium returns Π = δ Σ w_mkt where w_mkt = inverse-volatility weights
    Views: one absolute view per signaled asset, Q_i = dists[i].stats["close"]["mean"] / price - 1
    View uncertainty: Ω_ii ∝ dists[i].entropy() — high-entropy (uncertain) days get weak views
    Posterior: mu_BL blends prior Π and views Q based on confidence (Ω)

    The posterior expected returns are fed into the same max-Sharpe optimizer as MVOAllocator.

    Attributes:
        name: "black_litterman_allocator"
        tau: Confidence scaling parameter (scales prior impact). Default: 0.05.
        delta: Prior market risk premium parameter. Default: 2.5.
        lookback: Number of days of historical returns for covariance. Default: 120.
        gross_cap: Gross leverage cap (sum of |w_i|). Default: 1.0.
        max_weight: Maximum absolute weight for any single asset. Default: 0.35.
    """

    name: str = "black_litterman_allocator"

    def __init__(
        self,
        tau: float = 0.05,
        delta: float = 2.5,
        lookback: int = 120,
        gross_cap: float = 1.0,
        max_weight: float = 0.35,
    ):
        """
        Initialize Black-Litterman allocator.

        Args:
            tau: Confidence scaling parameter. Smaller tau makes prior more confident
                 (smaller Ω), moving posterior toward prior Π. Default: 0.05.
            delta: Prior market risk premium (multiplier for inverse-vol weights).
                  Default: 2.5.
            lookback: Number of days of historical returns for covariance estimation.
                     Default: 120 (approximately 6 months).
            gross_cap: Gross leverage cap (sum of absolute values of weights).
                      Default: 1.0.
            max_weight: Maximum absolute weight for any single asset.
                       Default: 0.35.

        Examples:
            >>> allocator = BlackLittermanAllocator(tau=0.05, delta=2.5, lookback=120)
            >>> weights = allocator.allocate(signals, returns, dists, context)
        """
        self.tau = tau
        self.delta = delta
        self.lookback = lookback
        self.gross_cap = gross_cap
        self.max_weight = max_weight

    def allocate(
        self,
        signals: Dict[str, "Signal"],
        returns: pd.DataFrame,
        dists: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute Black-Litterman optimal weights.

        Args:
            signals: Dict[symbol -> Signal] of active signals.
            returns: pd.DataFrame of (n_obs, n_assets) daily log returns.
                    Index: dates, Columns: asset symbols.
            dists: Dict[symbol -> KairosDistribution] predicted distributions per asset.
            context: Dict[str, Any] execution context (unused).

        Returns:
            Dict[symbol -> float] of target weights, signed (long/short).
            Sum of absolute values <= gross_cap.
            Individual absolute values <= max_weight.
            Signs match signal directions.

        Fall-back behavior:
            If len(returns) < self.min_obs (default 60), returns equal-weight allocation.
            If optimization fails for any reason, returns equal-weight allocation.

        Examples:
            >>> allocator = BlackLittermanAllocator()
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

        # Compute prior: w_mkt = inverse-volatility weights
        try:
            w_mkt = self._inverse_vol_weights(cov, signals)
        except Exception:
            return _fallback_equal_weight(signals)

        # Compute prior equilibrium returns: Π = δ Σ w_mkt
        try:
            pi = self.delta * (cov @ w_mkt)
        except Exception:
            return _fallback_equal_weight(signals)

        # Compute views and view uncertainty
        try:
            Q, Omega = self._compute_views_and_uncertainty(symbols, signals, dists, cov)
        except Exception:
            return _fallback_equal_weight(signals)

        # Compute posterior
        try:
            mu_bl = _bl_posterior(pi, Q, cov, Omega, tau=self.tau)
        except Exception:
            return _fallback_equal_weight(signals)

        # Solve max-Sharpe using posterior mu
        try:
            weights_dict = _max_sharpe_solve(
                symbols,
                mu_bl,
                cov,
                signals,
                gross_cap=self.gross_cap,
                max_weight=self.max_weight,
                rf=0.0,
            )
            return weights_dict
        except Exception:
            # Fallback to equal weight on solver failure
            return _fallback_equal_weight(signals)

    def _inverse_vol_weights(
        self, cov: np.ndarray, signals: Dict[str, "Signal"]
    ) -> np.ndarray:
        """
        Compute inverse-volatility weights for signaled assets.

        w_mkt_i = (1 / σ_i) / sum_j (1 / σ_j)

        Args:
            cov: Covariance matrix (n_assets, n_assets).
            signals: Dict[symbol -> Signal] for filtering active assets.

        Returns:
            Inverse-vol weight vector (n_assets,), summing to 1.
        """
        n = len(signals)
        vols = np.sqrt(np.diag(cov))

        if np.any(vols <= 0):
            raise ValueError("Non-positive volatility in covariance diagonal")

        inv_vols = 1.0 / vols
        w_mkt = inv_vols / np.sum(inv_vols)

        return w_mkt

    def _compute_views_and_uncertainty(
        self,
        symbols: List[str],
        signals: Dict[str, "Signal"],
        dists: Dict[str, Any],
        cov: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute views Q and view uncertainty Ω.

        Views: Q_i = dists[symbol].stats["close"]["mean"] / entry_price - 1
               (absolute return forecast)
        Uncertainty: Ω_ii = (entropy_i / ln(20)) * τ * σ_i^2
                   Scaled so max-entropy (ln 20 ≈ 3.0) views are weak (large Ω_ii)

        Args:
            symbols: List of asset symbols (order matching cov).
            signals: Dict[symbol -> Signal] for entry prices.
            dists: Dict[symbol -> KairosDistribution] for stats and entropy.
            cov: Covariance matrix (n_assets, n_assets).

        Returns:
            Q: View vector (n_assets,).
            Ω: View uncertainty matrix (n_assets, n_assets), diagonal.
        """
        n = len(symbols)
        Q = np.zeros(n)
        Omega = np.zeros((n, n))

        ln_20 = np.log(20.0)

        for i, sym in enumerate(symbols):
            signal = signals[sym]
            dist = dists[sym]
            current_price = signal.entry

            # View: expected relative return from distribution
            try:
                close_mean = dist.stats["close"]["mean"]
                Q[i] = close_mean / current_price - 1.0
            except (KeyError, TypeError):
                Q[i] = 0.0

            # View uncertainty: entropy-scaled
            try:
                entropy = dist.entropy()
            except Exception:
                entropy = 1.5  # Fallback if entropy unavailable

            # Ω_ii = (entropy / ln(20)) * τ * σ_i^2
            # When entropy → ln(20), Ω_ii → τ * σ_i^2 (weak view)
            # When entropy → 0, Ω_ii → 0 (strong/confident view)
            var_i = cov[i, i]
            if var_i <= 0:
                var_i = 1e-6
            omega_ii = (entropy / ln_20) * self.tau * var_i
            Omega[i, i] = max(omega_ii, 1e-10)  # Ensure numerical stability

        return Q, Omega


# =============================================================================
# EIGEN PORTFOLIO ALLOCATOR
# =============================================================================

def _eigen_portfolios(corr: np.ndarray, k: int) -> np.ndarray:
    """
    Extract top-k eigenvectors from correlation matrix, excluding PC1 (market mode).

    Performs eigendecomposition of correlation matrix, returning the next-k
    eigenvectors (indices 1 to k inclusive) — the dominant eigenvector (PC1)
    is excluded. This represents the variance explained by each orthogonal
    factor after removing the market mode.

    Args:
        corr: Correlation matrix (n_assets, n_assets), symmetric.
        k: Number of eigenvectors to return (k >= 1).

    Returns:
        Eigenvectors matrix of shape (n_assets, k), with columns as the next-k
        eigenvectors ranked by eigenvalue (largest first), excluding PC1.
        Columns are orthonormal (from np.linalg.eigh).

    Raises:
        ValueError: If k < 1 or k >= n_assets (need at least 2 components:
                   PC1 to exclude + at least one to return).

    Examples:
        >>> corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        >>> V = _eigen_portfolios(corr, 1)
        >>> V.shape
        (2, 1)
        >>> # V[:, 0] is the second-largest eigenvector
    """
    n = corr.shape[0]

    if k < 1 or k >= n:
        raise ValueError(f"k must be in [1, {n-1}], got {k}")

    # Eigendecompose correlation matrix
    # np.linalg.eigh returns eigenvalues in ascending order
    eigvals, eigvecs = np.linalg.eigh(corr)

    # Reverse to get descending order (largest eigenvalue first = PC1)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Return the next-k eigenvectors (indices 1 to k), excluding PC1 (index 0)
    return eigvecs[:, 1:k+1]


class EigenAllocator(PortfolioAllocator):
    """
    Eigen-Portfolio allocator using PCA on correlation matrix, excluding market mode.

    Performs PCA on the correlation matrix of trailing returns:
    1. Eigendecompose correlation matrix
    2. Drop PC1 (market mode, largest eigenvalue)
    3. Allocate to top-k remaining eigenvectors, weighted by their eigenvalues
    4. Project back to asset space and re-sign by signal direction

    This provides a market-neutral decomposition that reduces systematic risk
    by excluding the dominant common factor (market mode). The resulting
    weight vector is more orthogonal to the equal-weight basket than the
    full-market eigenvector would be.

    Attributes:
        name: "eigen_allocator"
        n_components: Number of eigenvectors to use (after excluding PC1).
                     Default: 3.
        lookback: Number of days of historical returns for correlation.
                 Default: 120 (approximately 6 months).
    """

    name: str = "eigen_allocator"

    def __init__(self, n_components: int = 3, lookback: int = 120):
        """
        Initialize Eigen allocator.

        Args:
            n_components: Number of eigenvectors to use (after excluding PC1).
                         Default: 3.
            lookback: Number of days of historical returns for correlation.
                     Default: 120 (approximately 6 months).

        Examples:
            >>> allocator = EigenAllocator(n_components=3, lookback=120)
            >>> weights = allocator.allocate(signals, returns, dists, context)
        """
        self.n_components = n_components
        self.lookback = lookback

    def allocate(self, signals: Dict[str, "Signal"],
                 returns: pd.DataFrame,
                 dists: Dict[str, Any],
                 context: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute eigen-portfolio optimal weights.

        Excludes the market mode (PC1) and allocates to the next-k eigenvectors
        weighted by their eigenvalues, then projects back to asset space.

        Args:
            signals: Dict[symbol -> Signal] of active signals.
            returns: pd.DataFrame of (n_obs, n_assets) daily log returns.
                    Index: dates, Columns: asset symbols.
            dists: Dict[symbol -> KairosDistribution] (unused for Eigen).
            context: Dict[str, Any] execution context (unused).

        Returns:
            Dict[symbol -> float] of target weights, signed (long/short).
            Sum of absolute values = 1.0 (fully invested after normalization).
            Signs match signal directions (LONG/SHORT/FLAT).

        Fall-back behavior:
            If len(returns) < self.min_obs (default 60), returns equal-weight.
            If fewer than 2 assets, returns equal-weight (no eigenvector decomposition).
            If optimization fails, returns equal-weight.

        Examples:
            >>> allocator = EigenAllocator()
            >>> weights = allocator.allocate(signals, returns, dists, {})
            >>> assert sum(abs(w) for w in weights.values()) <= 1.0 + 1e-6
        """
        # Check if enough observations
        if len(returns) < self.min_obs:
            return _fallback_equal_weight(signals)

        # Extract symbols
        symbols = list(signals.keys())
        if not symbols:
            return {}

        # Fallback: need at least 2 assets for meaningful eigenvector decomposition
        # (PC1 to exclude + at least one to return)
        if len(symbols) < 2:
            return _fallback_equal_weight(signals)

        # Get trailing returns for correlation estimation
        trailing_returns = returns[symbols].tail(self.lookback)
        if len(trailing_returns) < 2:
            return _fallback_equal_weight(signals)

        try:
            # Compute correlation matrix
            corr = np.corrcoef(trailing_returns.T)

            # Handle NaN in correlation (e.g., constant series)
            if np.isnan(corr).any():
                return _fallback_equal_weight(signals)

            # Determine how many components to use
            n = len(symbols)
            k = min(self.n_components, n - 1)

            if k < 1:
                # Fallback if not enough components
                return _fallback_equal_weight(signals)

            # Get next-k eigenvectors (excluding PC1)
            eigenvectors = _eigen_portfolios(corr, k)

            # Eigendecompose to get eigenvalues
            eigvals, eigvecs_full = np.linalg.eigh(corr)
            eigvals = eigvals[::-1]  # Descending order

            # Get the eigenvalues for the next-k components
            eigenvalues_used = eigvals[1:k+1]

            # Weight each eigenvector by its eigenvalue
            # Normalize eigenvalues to sum to 1 for weighting
            if np.sum(eigenvalues_used) <= 0:
                return _fallback_equal_weight(signals)

            weights_eig = eigenvalues_used / np.sum(eigenvalues_used)

            # Combine eigenvectors: weighted sum of eigenvectors
            # Portfolio weight = sum_i (weight_i * eigenvector_i)
            portfolio_weights = np.zeros(n)
            for i in range(k):
                portfolio_weights += weights_eig[i] * eigenvectors[:, i]

            # Extract magnitudes (eigenvectors have arbitrary signs, so use absolute values)
            magnitudes = np.abs(portfolio_weights)

            # Normalize magnitudes to sum to 1
            total_mag = np.sum(magnitudes)
            if total_mag <= 0:
                return _fallback_equal_weight(signals)

            magnitudes = magnitudes / total_mag

            # Apply signal directions and assemble final weights
            weights = {}
            for j, sym in enumerate(symbols):
                mag = magnitudes[j]  # Always positive
                direction = signals[sym].direction

                # Extract direction value (handles both enum and raw int)
                if hasattr(direction, 'value'):
                    dir_val = direction.value
                else:
                    dir_val = direction

                if dir_val == 1:  # LONG
                    w = float(mag)
                elif dir_val == -1:  # SHORT
                    w = float(-mag)
                else:  # FLAT
                    w = 0.0

                if abs(w) < 1e-10:
                    w = 0.0
                weights[sym] = w

            return weights

        except Exception:
            # Fallback to equal weight on any computation error
            return _fallback_equal_weight(signals)


# =============================================================================
# UNIVERSAL PORTFOLIO ALLOCATOR (COVER)
# =============================================================================

class UniversalAllocator(PortfolioAllocator):
    """
    Universal Portfolio allocator (Cover 1991): wealth-weighted mixture of CRPs.

    Maintains a Dirichlet grid of constant-rebalanced portfolios (CRPs) over the
    signaled assets and tracks cumulative wealth per grid point. On each call,
    the allocator updates the wealth of each grid point using realized returns,
    then outputs the wealth-weighted mixture of grid points (normalized and
    re-signed per signal direction).

    This stateful allocator learns online which portfolio combinations performed
    best over the rolling window. The wealth-weighting mechanism automatically
    gives more weight to grid points that have generated higher cumulative returns.

    **State management:**
    - Grid is regenerated when the symbol set changes (detecting universe drift).
    - Wealth is reset to 1.0 when grid is regenerated.
    - reset() method clears all state for walk-forward folds.

    Attributes:
        name: "universal_allocator"
        grid_step: Resolution of the Dirichlet grid (default 0.1).
                  Weights take values in {0, 0.1, 0.2, ..., 1.0}.
        grid_symbols: Current symbol set (None if not yet initialized).
        grid: List of constant-rebalanced portfolio weight vectors.
        wealth: Array of cumulative wealth per grid point.
    """

    name: str = "universal_allocator"

    def __init__(self, grid_step: float = 0.1):
        """
        Initialize Universal allocator.

        Args:
            grid_step: Resolution of the Dirichlet grid. Default: 0.1.
                      With 3 assets and grid_step=0.1, generates 66 CRPs.

        Examples:
            >>> allocator = UniversalAllocator(grid_step=0.1)
            >>> weights = allocator.allocate(signals, returns, dists, context)
        """
        self.grid_step = float(grid_step)
        self.grid = None
        self.grid_symbols = None
        self.wealth = None

    def allocate(
        self,
        signals: Dict[str, "Signal"],
        returns: pd.DataFrame,
        dists: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Allocate using wealth-weighted mixture of constant-rebalanced portfolios.

        On each call:
        1. Check if symbol set has changed; regenerate grid if needed (reset wealth to 1)
        2. Update wealth for each grid point using today's log returns
        3. Compute wealth-weighted average allocation
        4. Apply signal directions and return as dict

        Args:
            signals: Dict[symbol -> Signal] of active signals.
            returns: pd.DataFrame of (n_obs, n_assets) daily log returns.
                    Index: dates, Columns: asset symbols.
                    Last row is used for today's return update.
            dists: Dict[symbol -> KairosDistribution] (unused for Universal).
            context: Dict[str, Any] execution context (unused).

        Returns:
            Dict[symbol -> float] of target weights, signed (long/short).
            Sum of absolute values = 1.0 (fully invested after normalization).
            Signs match signal directions (LONG/SHORT/FLAT).

        Fall-back behavior:
            If len(returns) < self.min_obs (default 60), returns equal-weight.
            If no signals, returns empty dict.

        Examples:
            >>> allocator = UniversalAllocator()
            >>> weights = allocator.allocate(signals, returns, dists, {})
            >>> assert sum(abs(w) for w in weights.values()) <= 1.0 + 1e-6
        """
        symbols = list(signals.keys())

        # Check if enough observations
        if len(returns) < self.min_obs:
            return _fallback_equal_weight(signals)

        if not symbols:
            return {}

        # Check if symbol set changed; regenerate grid if needed
        if self.grid is None or set(symbols) != set(self.grid_symbols):
            self._regenerate_grid(symbols)

        # Get today's returns (last row of the DataFrame)
        today_returns = returns[symbols].iloc[-1].values  # numpy array

        # Update wealth for each grid point: wealth_i *= (1 + grid_weights_i · returns)
        for i, grid_weights in enumerate(self.grid):
            daily_return = float(np.dot(grid_weights, today_returns))
            self.wealth[i] *= (1.0 + daily_return)

        # Compute wealth-weighted mixture
        total_wealth = np.sum(self.wealth)
        if total_wealth <= 0:
            # All wealth wiped out; reset and fall back to equal weight
            self._regenerate_grid(symbols)
            return _fallback_equal_weight(signals)

        wealth_weights = self.wealth / total_wealth

        # Average the grid points weighted by their wealth
        avg_allocation = np.zeros(len(symbols))
        for i, grid_weights in enumerate(self.grid):
            avg_allocation += wealth_weights[i] * grid_weights

        # Apply signal directions
        weights = {}
        for j, sym in enumerate(symbols):
            mag = float(avg_allocation[j])
            direction = signals[sym].direction

            # Extract direction value (handles both enum and raw int)
            if hasattr(direction, 'value'):
                dir_val = direction.value
            else:
                dir_val = direction

            if dir_val == 1:  # LONG
                w = float(mag)
            elif dir_val == -1:  # SHORT
                w = float(-mag)
            else:  # FLAT
                w = 0.0

            if abs(w) < 1e-10:
                w = 0.0
            weights[sym] = w

        return weights

    def reset(self) -> None:
        """
        Reset all internal state for walk-forward folds.

        Clears the grid and wealth array so the next call starts fresh.
        """
        self.grid = None
        self.grid_symbols = None
        self.wealth = None

    def _regenerate_grid(self, symbols: List[str]) -> None:
        """
        Generate all constant-rebalanced portfolio combinations.

        Creates a Dirichlet grid of all weight vectors where:
        - Each component is a multiple of self.grid_step
        - All components sum to exactly 1.0
        - For n assets and step=0.1: generates C(n + num_steps - 1, num_steps) vectors

        Args:
            symbols: List of asset symbols (order is preserved).

        Sets:
            self.grid: List of numpy arrays, each of shape (n_assets,)
            self.grid_symbols: Stores symbols for future change detection
            self.wealth: Initializes wealth to 1.0 for each grid point
        """
        n = len(symbols)
        num_steps = int(round(1.0 / self.grid_step))

        grid = []

        # Recursive generator for all non-negative integer partitions
        # summing to num_steps (representing multiples of grid_step)
        def generate_partitions(n_assets, remaining_steps, current_partition):
            """
            Generate all partitions of remaining_steps into n_assets non-negative integers.

            Args:
                n_assets: Number of assets still to allocate.
                remaining_steps: Sum of integers to allocate.
                current_partition: List of weights accumulated so far.
            """
            if n_assets == 1:
                # Last asset gets all remaining steps
                final_partition = current_partition + [remaining_steps]
                grid.append(np.array(final_partition, dtype=float) * self.grid_step)
            else:
                # Try each value from 0 to remaining_steps for this asset
                for steps_here in range(remaining_steps + 1):
                    generate_partitions(
                        n_assets - 1,
                        remaining_steps - steps_here,
                        current_partition + [steps_here]
                    )

        generate_partitions(n, num_steps, [])

        self.grid = grid
        self.grid_symbols = symbols
        self.wealth = np.ones(len(grid), dtype=float)


# =============================================================================
# GENETIC ALGORITHM ALLOCATOR
# =============================================================================

class GAAllocator(PortfolioAllocator):
    """
    Genetic Algorithm allocator: evolves weight vectors to maximize trailing Sharpe.

    Uses a population-based evolutionary algorithm to find optimal portfolio weights:
    - Fitness: trailing Sharpe ratio of portfolio daily returns
    - Population: 50 normalized non-negative magnitude vectors
    - Selection: tournament selection (k=3)
    - Crossover: blend crossover with uniform alpha
    - Mutation: Gaussian with sigma=0.05, clip to >=0, renormalize
    - Elitism: keep best 2 individuals
    - Generations: 20

    Caching & determinism:
    - RNG seeded deterministically from date (YYYYMMDD format)
    - Result cached and only re-run when refit_days new calls elapsed
    - Tracks call counter and exposes run_count attribute
    - reset() clears all state for walk-forward folds

    Caps & signs:
    - Uniform scale all weights to respect gross_cap (sum of |w| <= gross_cap)
    - Clip individual weights to max_weight, then renormalize
    - Apply signal-direction signs (LONG/SHORT/FLAT)
    - Fallback to equal-weight when observations < min_obs

    Attributes:
        name: "ga_allocator"
        lookback: Trailing window for Sharpe calculation (default 60 days)
        population: Population size (default 50)
        generations: Number of evolution generations (default 20)
        mutation_sigma: Gaussian mutation std dev (default 0.05)
        gross_cap: Gross leverage cap (default 1.0)
        max_weight: Individual position cap (default 0.35)
        refit_days: Number of days between re-runs (default 5)
        last_fitness_history: List of best fitness per generation from last run
        run_count: Number of times allocate() has been called
    """

    name: str = "ga_allocator"

    def __init__(
        self,
        lookback: int = 60,
        population: int = 50,
        generations: int = 20,
        mutation_sigma: float = 0.05,
        gross_cap: float = 1.0,
        max_weight: float = 0.35,
        refit_days: int = 5,
    ):
        """
        Initialize GA allocator.

        Args:
            lookback: Trailing window for Sharpe ratio (default 60 days).
            population: Population size (default 50).
            generations: Number of evolution generations (default 20).
            mutation_sigma: Gaussian mutation std dev (default 0.05).
            gross_cap: Gross leverage cap, sum of |w_i| (default 1.0).
            max_weight: Individual position cap (default 0.35).
            refit_days: Days between refits; re-run every refit_days calls
                       (default 5).

        Examples:
            >>> allocator = GAAllocator(lookback=60, population=50, generations=20)
            >>> weights = allocator.allocate(signals, returns, dists, context)
        """
        self.lookback = lookback
        self.population = population
        self.generations = generations
        self.mutation_sigma = mutation_sigma
        self.gross_cap = gross_cap
        self.max_weight = max_weight
        self.refit_days = refit_days

        # State tracking
        self.last_fitness_history = []
        self.run_count = 0
        self._cache_date = None
        self._cached_weights = None
        self._cached_symbols = None

    def reset(self) -> None:
        """
        Reset all internal state for walk-forward folds.

        Clears cache and counters so next call starts fresh.
        """
        self.last_fitness_history = []
        self.run_count = 0
        self._cache_date = None
        self._cached_weights = None
        self._cached_symbols = None

    def allocate(
        self,
        signals: Dict[str, "Signal"],
        returns: pd.DataFrame,
        dists: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Allocate using GA-optimized weights.

        Runs the GA to maximize trailing Sharpe if:
        - This is the first call, OR
        - refit_days have elapsed since last run, OR
        - Symbol set has changed

        Otherwise, returns cached weights.

        Args:
            signals: Dict[symbol -> Signal] of active signals.
            returns: pd.DataFrame of (n_obs, n_assets) daily log returns.
                    Index: dates, Columns: asset symbols.
            dists: Dict[symbol -> KairosDistribution] (unused for GA).
            context: Dict[str, Any]; uses context.get("current_date") if present
                    for deterministic seed, else uses returns index last element.

        Returns:
            Dict[symbol -> float] of target weights, signed (long/short).
            Sum of absolute values <= gross_cap.
            Individual absolute values <= max_weight.
            Signs match signal directions.

        Fall-back behavior:
            If len(returns) < self.min_obs (default 60), returns equal-weight.
            If GA fails, returns equal-weight.

        Examples:
            >>> allocator = GAAllocator()
            >>> weights = allocator.allocate(signals, returns, dists, context)
        """
        # Increment call counter
        self.run_count += 1

        # Check if enough observations
        if len(returns) < self.min_obs:
            return _fallback_equal_weight(signals)

        symbols = list(signals.keys())
        if not symbols:
            return {}

        # Determine the date for caching and seed
        if "current_date" in context:
            current_date = context["current_date"]
        else:
            current_date = returns.index[-1]

        # Check if we should use cache
        if (
            self._cache_date is not None
            and self._cached_symbols == symbols
            and (current_date - self._cache_date).days < self.refit_days
        ):
            # Return cached weights
            return self._cached_weights.copy()

        # Need to run GA: extract trailing returns
        trailing_returns = returns[symbols].tail(self.lookback)
        if len(trailing_returns) < 2:
            return _fallback_equal_weight(signals)

        # Run GA
        try:
            magnitudes = self._run_ga(trailing_returns, symbols)
        except Exception:
            return _fallback_equal_weight(signals)

        # Apply caps and signs
        try:
            weights_dict = self._apply_caps_and_signs(magnitudes, symbols, signals)
        except Exception:
            return _fallback_equal_weight(signals)

        # Update cache
        self._cache_date = current_date
        self._cached_weights = weights_dict.copy()
        self._cached_symbols = symbols

        return weights_dict

    def _run_ga(self, returns: pd.DataFrame, symbols: List[str]) -> np.ndarray:
        """
        Run genetic algorithm to maximize Sharpe ratio.

        Args:
            returns: DataFrame of trailing returns (lookback rows, one per symbol).
            symbols: Asset symbols (for seeding and logging).

        Returns:
            Magnitudes array of shape (n_assets,), non-negative and normalized to sum to 1.

        Raises:
            ValueError: If GA fails to initialize or evolve.
        """
        n_assets = len(symbols)
        returns_array = returns.values  # (lookback, n_assets)

        # Deterministic seed from last date in returns
        seed_date = returns.index[-1]
        seed_val = int(seed_date.strftime("%Y%m%d"))
        rng = np.random.default_rng(seed_val)

        # Initialize population: random normalized non-negative vectors
        population = []
        for _ in range(self.population):
            # Random non-negative magnitudes
            mag = rng.exponential(1.0, n_assets)
            # Normalize to sum to 1
            mag = mag / np.sum(mag)
            population.append(mag)

        population = np.array(population)

        # Track fitness history
        fitness_history = []

        # Evolution loop
        for gen in range(self.generations):
            # Evaluate fitness for each individual
            fitness = np.array([
                self._compute_fitness(ind, returns_array)
                for ind in population
            ])

            # Track best fitness
            best_fitness = np.max(fitness)
            fitness_history.append(best_fitness)

            # Selection: tournament selection (k=3)
            selected_indices = []
            for _ in range(self.population - 2):  # Leave 2 for elitism
                # Tournament: pick k random individuals, select best
                tournament_idx = rng.choice(self.population, size=3, replace=False)
                tournament_fitness = fitness[tournament_idx]
                winner_idx = tournament_idx[np.argmax(tournament_fitness)]
                selected_indices.append(winner_idx)

            # Crossover & mutation: create new population
            new_population = []

            # Elitism: keep best 2
            best_two_idx = np.argsort(fitness)[-2:]
            for idx in best_two_idx:
                new_population.append(population[idx].copy())

            # Create offspring via crossover and mutation
            for _ in range(self.population - 2):
                # Blend crossover: pick two parents, blend with uniform alpha
                parent_idx = rng.choice(selected_indices, size=2, replace=False)
                parent1 = population[parent_idx[0]]
                parent2 = population[parent_idx[1]]

                # Blend crossover: alpha uniform from [0, 1]
                alpha = rng.uniform(0.0, 1.0)
                child = alpha * parent1 + (1.0 - alpha) * parent2

                # Mutation: add Gaussian noise, clip to >=0, renormalize
                noise = rng.normal(0.0, self.mutation_sigma, n_assets)
                child = child + noise
                child = np.maximum(child, 0.0)  # Clip to >=0
                if np.sum(child) > 0:
                    child = child / np.sum(child)  # Renormalize
                else:
                    # Degenerate case: reinitialize
                    child = rng.exponential(1.0, n_assets)
                    child = child / np.sum(child)

                new_population.append(child)

            population = np.array(new_population)

        # Store fitness history for test introspection
        self.last_fitness_history = fitness_history

        # Return best individual from final population
        final_fitness = np.array([
            self._compute_fitness(ind, returns_array)
            for ind in population
        ])
        best_idx = np.argmax(final_fitness)
        return population[best_idx]

    def _compute_fitness(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute Sharpe ratio fitness for a weight vector.

        Sharpe = mean(portfolio_returns) / std(portfolio_returns) * sqrt(252)
        where portfolio_returns = returns @ weights (daily log returns)

        Args:
            weights: Weight vector (n_assets,), assumed normalized to sum to 1.
            returns: Returns array (n_obs, n_assets).

        Returns:
            Sharpe ratio (float). If std=0, returns -inf.
        """
        portfolio_returns = np.dot(returns, weights)  # (n_obs,)
        mean_ret = np.mean(portfolio_returns)
        std_ret = np.std(portfolio_returns, ddof=1)

        if std_ret <= 0:
            return float('-inf')

        sharpe = (mean_ret / std_ret) * np.sqrt(252)
        return sharpe

    def _apply_caps_and_signs(
        self,
        magnitudes: np.ndarray,
        symbols: List[str],
        signals: Dict[str, "Signal"],
    ) -> Dict[str, float]:
        """
        Apply gross_cap and max_weight constraints, then sign by directions.

        Steps:
        1. Uniform scale to respect gross_cap (sum of |w| <= gross_cap)
        2. Clip individual |w_i| to max_weight
        3. Scale down uniformly again if needed to respect gross_cap after clipping
        4. Apply signal-direction signs

        Args:
            magnitudes: Non-negative weight magnitudes (sum to 1).
            symbols: Asset symbols (order matching magnitudes).
            signals: Dict[symbol -> Signal] for directions.

        Returns:
            Dict[symbol -> float] of signed weights.
        """
        n = len(symbols)

        # Step 1: Uniform scale to respect gross_cap
        total_mag = np.sum(magnitudes)
        if total_mag > self.gross_cap:
            magnitudes = magnitudes * (self.gross_cap / total_mag)

        # Step 2: Clip individual weights to max_weight (hard cap)
        magnitudes = np.minimum(magnitudes, self.max_weight)

        # Step 3: If sum now exceeds gross_cap, scale down uniformly
        total_after_clip = np.sum(magnitudes)
        if total_after_clip > self.gross_cap:
            magnitudes = magnitudes * (self.gross_cap / total_after_clip)

        # Step 4: Apply signal directions
        weights = {}
        for i, sym in enumerate(symbols):
            mag = magnitudes[i]
            direction = signals[sym].direction

            # Extract direction value
            if hasattr(direction, 'value'):
                dir_val = direction.value
            else:
                dir_val = direction

            if dir_val == 1:  # LONG
                w = float(mag)
            elif dir_val == -1:  # SHORT
                w = float(-mag)
            else:  # FLAT
                w = 0.0

            if abs(w) < 1e-10:
                w = 0.0
            weights[sym] = w

        return weights
