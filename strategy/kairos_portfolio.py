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
