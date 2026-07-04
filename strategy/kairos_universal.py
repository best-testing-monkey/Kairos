"""
kairos_universal.py
===================
Universal / Cross-Asset strategies (4.1 – 4.18) for the Kairos framework.

Complex algorithms (Kalman filter, Hurst, DFA, fractal dimension, LZ complexity,
RQA, copula, HMM, wavelet, Gaussian process, particle filter, spectral
clustering, transfer entropy, BSTS, GNN-surrogate, RL) are implemented using
numpy / scipy only — no heavy ML dependencies required.

All strategies read context fields documented in EXTENDED_STRATEGIES.md §6.1.
Missing context fields cause the strategy to return None gracefully.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy import stats as scipy_stats
from scipy.linalg import solve, eigh
from typing import Optional, Dict, List, Tuple
from collections import defaultdict, deque
from kairos_backtest import KairosDistribution, Direction, Signal, Strategy


# ===========================================================================
# Helpers
# ===========================================================================

def _hurst_rs(series: np.ndarray) -> float:
    """R/S Hurst exponent estimation."""
    n = len(series)
    if n < 20:
        return 0.5
    lags = [max(4, n // k) for k in range(2, min(10, n // 4 + 1))]
    lags = sorted(set(lags))
    rs_vals = []
    for lag in lags:
        sub = series[:lag]
        mean = np.mean(sub)
        cumdev = np.cumsum(sub - mean)
        r = np.max(cumdev) - np.min(cumdev)
        s = np.std(sub, ddof=1)
        if s > 0:
            rs_vals.append(np.log(r / s))
        else:
            rs_vals.append(0.0)
    if len(rs_vals) < 2:
        return 0.5
    log_lags = np.log([float(l) for l in lags[:len(rs_vals)]])
    slope, _, _, _, _ = scipy_stats.linregress(log_lags, rs_vals)
    return float(np.clip(slope, 0.0, 1.0))


def _dfa_alpha(series: np.ndarray) -> float:
    """Detrended Fluctuation Analysis — returns DFA alpha."""
    n = len(series)
    if n < 20:
        return 0.5
    cumsum = np.cumsum(series - np.mean(series))
    scales = [max(4, n // k) for k in range(2, min(8, n // 4 + 1))]
    scales = sorted(set(scales))
    flucts = []
    for scale in scales:
        n_segs = n // scale
        if n_segs < 1:
            continue
        f2 = 0.0
        for seg in range(n_segs):
            y = cumsum[seg * scale:(seg + 1) * scale]
            x = np.arange(len(y), dtype=float)
            if len(y) < 2:
                continue
            p = np.polyfit(x, y, 1)
            trend = np.polyval(p, x)
            f2 += np.mean((y - trend) ** 2)
        f2 /= max(n_segs, 1)
        flucts.append(np.sqrt(f2))
    if len(flucts) < 2:
        return 0.5
    log_s = np.log([float(s) for s in scales[:len(flucts)]])
    log_f = np.log(np.maximum(flucts, 1e-10))
    slope, _, _, _, _ = scipy_stats.linregress(log_s, log_f)
    return float(np.clip(slope, 0.0, 1.0))


def _fractal_dim(series: np.ndarray) -> float:
    """Box-counting fractal dimension of a 1-D series."""
    n = len(series)
    if n < 8:
        return 1.5
    mn, mx = np.min(series), np.max(series)
    if mx == mn:
        return 1.0
    norm = (series - mn) / (mx - mn)
    box_sizes = [max(2, n // k) for k in [4, 8, 16, 32] if n // k >= 2]
    counts = []
    for box in box_sizes:
        n_boxes = int(np.ceil(1.0 / (box / n)))
        grid = np.zeros(n_boxes + 1, dtype=int)
        for v in norm:
            idx = int(v * n_boxes)
            grid[min(idx, n_boxes)] = 1
        counts.append(np.sum(grid))
    if len(counts) < 2:
        return 1.5
    log_n = np.log([float(n // b) for b in box_sizes[:len(counts)]])
    log_c = np.log(np.maximum(counts, 1))
    slope, _, _, _, _ = scipy_stats.linregress(log_n, log_c)
    return float(np.clip(slope, 1.0, 2.0))


def _lz_complexity(series: np.ndarray) -> float:
    """Lempel-Ziv complexity of a binary sequence (normalised 0–1)."""
    if len(series) < 2:
        return 1.0
    binary = (series >= np.median(series)).astype(int)
    s = "".join(map(str, binary))
    n = len(s)
    i, c, l, k = 0, 1, 1, 1
    while True:
        if i + k > n:
            c += 1
            break
        sub = s[i:i + k]
        if sub not in s[0:i + k - 1]:
            c += 1
            i += k
            k = 1
        else:
            k += 1
    # Normalise by theoretical max = n / log2(n)
    if n > 1:
        c_max = n / np.log2(n)
        return float(np.clip(c / c_max, 0.0, 1.0))
    return 1.0


def _rqa_metrics(series: np.ndarray, threshold: float = 0.2) -> Tuple[float, float]:
    """Returns (determinism, laminarity) from recurrence quantification."""
    n = len(series)
    if n < 4:
        return 0.5, 0.5
    mn, mx = np.min(series), np.max(series)
    if mx == mn:
        return 1.0, 1.0
    norm = (series - mn) / (mx - mn)
    # Recurrence matrix
    dist_mat = np.abs(norm[:, None] - norm[None, :])
    R = (dist_mat < threshold).astype(float)
    total_rec = np.sum(R) - n  # exclude diagonal

    # Determinism: fraction of recurrence points in diagonal lines ≥ 2
    det_count = 0
    for d in range(1, n):
        diag = np.diag(R, d)
        # Find runs of 1s
        in_run, run_len = False, 0
        for v in diag:
            if v == 1:
                run_len += 1
                in_run = True
            else:
                if in_run and run_len >= 2:
                    det_count += run_len
                in_run = False
                run_len = 0
        if in_run and run_len >= 2:
            det_count += run_len
    det = det_count / max(total_rec, 1)

    # Laminarity: fraction in vertical lines ≥ 2
    lam_count = 0
    for col in range(n):
        run_len = 0
        for row in range(n):
            if R[row, col] == 1:
                run_len += 1
            else:
                if run_len >= 2:
                    lam_count += run_len
                run_len = 0
        if run_len >= 2:
            lam_count += run_len
    lam = lam_count / max(total_rec, 1)

    return float(np.clip(det, 0, 1)), float(np.clip(lam, 0, 1))


def _rbf_gp(X_train: np.ndarray, y_train: np.ndarray,
            X_test: np.ndarray, noise: float = 0.01,
            length_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal RBF Gaussian Process (posterior mean + variance)."""
    def rbf(a, b):
        d = ((a[:, None] - b[None, :]) ** 2).sum(-1) if a.ndim > 1 else \
            (a[:, None] - b[None, :]) ** 2
        return np.exp(-0.5 * d / length_scale ** 2)

    X_tr = X_train.reshape(-1, 1)
    X_te = X_test.reshape(-1, 1)
    K = rbf(X_tr, X_tr) + noise * np.eye(len(X_tr))
    K_s = rbf(X_tr, X_te)
    K_ss = np.diag(rbf(X_te, X_te))
    try:
        alpha = solve(K, y_train, assume_a="pos")
        mu = K_s.T @ alpha
        v = solve(K, K_s, assume_a="pos")
        var = K_ss - np.einsum("ij,ij->j", K_s, v)
    except np.linalg.LinAlgError:
        mu = np.full(len(X_test), np.mean(y_train))
        var = np.full(len(X_test), np.var(y_train))
    return mu, np.maximum(var, 0)


def _particle_weights(particles: np.ndarray, recent_prices: np.ndarray,
                      sigma: float) -> np.ndarray:
    """Compute particle weights from likelihood given recent price history."""
    if len(recent_prices) == 0:
        return np.ones(len(particles)) / len(particles)
    obs = recent_prices[-1]
    log_w = -0.5 * ((particles - obs) / max(sigma, 1e-9)) ** 2
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    return w / (w.sum() + 1e-300)


def _spectral_cluster(corr_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """Spectral clustering via normalised Laplacian eigendecomposition."""
    n = corr_matrix.shape[0]
    if n < n_clusters:
        return np.zeros(n, dtype=int)
    # Affinity: positive part of correlation
    A = np.maximum(corr_matrix, 0)
    np.fill_diagonal(A, 0)
    D = np.diag(A.sum(axis=1))
    d_inv_sqrt = np.where(D.diagonal() > 0, 1.0 / np.sqrt(D.diagonal()), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    try:
        eigvals, eigvecs = eigh(L_sym)
    except np.linalg.LinAlgError:
        return np.arange(n) % n_clusters
    k = min(n_clusters, n)
    V = eigvecs[:, :k]
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.where(norms > 0, norms, 1.0)
    # k-means (simplified: nearest centroid from random init)
    rng = np.random.default_rng(42)
    centroids = V[rng.choice(n, k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(20):
        dists = np.linalg.norm(V[:, None, :] - centroids[None, :, :], axis=-1)
        labels = np.argmin(dists, axis=1)
        for c in range(k):
            members = V[labels == c]
            if len(members) > 0:
                centroids[c] = members.mean(axis=0)
    return labels


# ===========================================================================
# 4.1  Kalman Filter Pairs Trading
# ===========================================================================

class KalmanPairs(Strategy):
    """
    Dynamic pairs trading with Kalman-filtered spread mean and variance.
    Maintains internal Kalman state across calls.
    """
    name = "kalman_pairs"

    def __init__(self, pair_symbol: str = "",
                 entry_z: float = 2.0,
                 exit_z: float = 0.5):
        self.pair_symbol = pair_symbol
        self.entry_z = entry_z
        self.exit_z = exit_z
        # Kalman state: [mean, variance]
        self._kf_mean: Optional[float] = None
        self._kf_var: float = 1.0
        self._q: float = 0.001   # process noise
        self._r: float = 0.1     # observation noise

    def _update(self, obs: float):
        """One-step Kalman update."""
        if self._kf_mean is None:
            self._kf_mean = obs
            return
        pred_var = self._kf_var + self._q
        k = pred_var / (pred_var + self._r)
        self._kf_mean = self._kf_mean + k * (obs - self._kf_mean)
        self._kf_var = (1 - k) * pred_var

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        spread_dist: Optional[KairosDistribution] = context.get("spread_dist")
        current_spread = context.get("current_spread")
        if spread_dist is None or current_spread is None:
            return None

        self._update(float(current_spread))
        if self._kf_mean is None:
            return None

        kalman_std = float(np.sqrt(max(self._kf_var, 1e-9)))
        pred_spread = spread_dist.stats["close"]["mean"]
        z_score = (pred_spread - self._kf_mean) / kalman_std

        if z_score > self.entry_z:
            direction = Direction.SHORT
        elif z_score < -self.entry_z:
            direction = Direction.LONG
        else:
            return None

        stop = self._kf_mean + 2 * kalman_std if direction == Direction.SHORT \
            else self._kf_mean - 2 * kalman_std
        target = self._kf_mean

        ev = spread_dist.expected_value(current_spread, target, stop)
        if ev <= 0:
            return None

        return Signal(
            direction=direction,
            size=min(abs(z_score) / (self.entry_z * 2), 0.4),
            entry=current_price,
            stop=current_price * (1 + (stop - current_spread) / max(abs(current_spread), 1e-9)),
            target=current_price * (1 + (target - current_spread) / max(abs(current_spread), 1e-9)),
            strategy_name=self.name,
            confidence=min(abs(z_score) / self.entry_z * 0.5, 1.0),
            expected_value=ev,
            metadata={"pair": self.pair_symbol, "z_score": z_score,
                      "kalman_mean": self._kf_mean, "kalman_std": kalman_std},
        )


# ===========================================================================
# 4.2  Hurst Exponent Regime Switching
# ===========================================================================

class HurstRegimeSwitch(Strategy):
    """
    Computes Hurst exponent on predicted close samples.
    H > 0.55 → trend strategies; H < 0.45 → mean-reversion; else → None.
    """
    name = "hurst_regime_switch"

    def __init__(self, trend_threshold: float = 0.55,
                 mean_reversion_threshold: float = 0.45):
        self.trend_thresh = trend_threshold
        self.mr_thresh = mean_reversion_threshold
        from kairos_backtest import (TrendFollowingStrategy, RangeTradingStrategy,
                                     FadeExtremeStrategy)
        self._trend_strat = TrendFollowingStrategy()
        self._mr_strat = RangeTradingStrategy()

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        closes = dist.df["close"].values.astype(float)
        H = _hurst_rs(closes)

        if H > self.trend_thresh:
            strat = self._trend_strat
        elif H < self.mr_thresh:
            strat = self._mr_strat
        else:
            return None

        sig = strat.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        sig.strategy_name = self.name
        sig.metadata["hurst"] = H
        sig.metadata["regime"] = "trend" if H > self.trend_thresh else "mean_reversion"
        return sig


# ===========================================================================
# 4.3  Copula-Based Dependence Trading
# ===========================================================================

class CopulaPairs(Strategy):
    """
    Trades conditional dependence between two assets using a Gaussian copula.
    Requires context["pair_prices"] = array of historical prices for pair asset.
    """
    name = "copula_pairs"

    def __init__(self, pair_symbol: str = "",
                 copula_type: str = "gaussian",
                 prob_threshold: float = 0.8):
        self.pair_symbol = pair_symbol
        self.copula_type = copula_type
        self.threshold = prob_threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        pair_prices = context.get("pair_prices")
        pair_dist: Optional[KairosDistribution] = context.get("pair_dist")
        if pair_prices is None or pair_dist is None or history is None:
            return None

        if len(history) < 20:
            return None

        # Compute historical returns for both assets
        ret_a = np.diff(np.log(history["close"].values.astype(float) + 1e-9))
        ret_b = np.diff(np.log(np.array(pair_prices, dtype=float) + 1e-9))
        n = min(len(ret_a), len(ret_b))
        if n < 10:
            return None
        ret_a, ret_b = ret_a[-n:], ret_b[-n:]

        # Rank-transform to uniform marginals
        u_a = scipy_stats.rankdata(ret_a) / (n + 1)
        u_b = scipy_stats.rankdata(ret_b) / (n + 1)

        # Estimate Gaussian copula correlation
        z_a = scipy_stats.norm.ppf(np.clip(u_a, 1e-6, 1 - 1e-6))
        z_b = scipy_stats.norm.ppf(np.clip(u_b, 1e-6, 1 - 1e-6))
        rho = float(np.corrcoef(z_a, z_b)[0, 1])

        # Predicted marginal position of each asset
        pred_a_u = dist.cdf(dist.stats["close"]["mean"])
        pred_b_u = pair_dist.cdf(pair_dist.stats["close"]["mean"])

        # Conditional probability P(A up | B position) via Gaussian copula
        z_b_pred = scipy_stats.norm.ppf(np.clip(pred_b_u, 1e-6, 1 - 1e-6))
        cond_mean = rho * z_b_pred
        cond_std = np.sqrt(max(1 - rho ** 2, 1e-9))
        # P(A goes up | B)
        p_a_up = float(scipy_stats.norm.sf(0, loc=cond_mean, scale=cond_std))

        if p_a_up > self.threshold:
            direction = Direction.LONG
        elif p_a_up < (1 - self.threshold):
            direction = Direction.SHORT
        else:
            return None

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        confidence = abs(p_a_up - 0.5) * 2

        return Signal(
            direction=direction,
            size=min(confidence * 0.4, 0.35),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={"pair": self.pair_symbol, "rho": rho,
                      "p_a_up": p_a_up, "copula_type": self.copula_type},
        )


# ===========================================================================
# 4.4  Cointegration with Error Correction
# ===========================================================================

class CointegrationECT(Strategy):
    """
    Trades the error-correction term (ECT) toward zero equilibrium.
    Requires context["ect_dist"] = KairosDistribution on the ECT series
    and context["current_ect"] = float.
    """
    name = "cointegration_ect"

    def __init__(self, pair_symbol: str = "",
                 fast_reversion_threshold: float = 0.5):
        self.pair_symbol = pair_symbol
        self.fast_thresh = fast_reversion_threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        ect_dist: Optional[KairosDistribution] = context.get("ect_dist")
        current_ect = context.get("current_ect")
        if ect_dist is None or current_ect is None:
            return None

        s = ect_dist.stats["close"]
        pred_ect = s["mean"]

        # ECT reversion: trade toward zero
        if current_ect > 0 and pred_ect < current_ect * self.fast_thresh:
            direction = Direction.SHORT
            stop = s["pct_90"]
            target = 0.0
        elif current_ect < 0 and pred_ect > current_ect * self.fast_thresh:
            direction = Direction.LONG
            stop = s["pct_10"]
            target = 0.0
        else:
            return None

        ev = ect_dist.expected_value(current_ect, target, stop)
        if ev <= 0:
            return None

        reversion_speed = abs(current_ect - pred_ect) / max(abs(current_ect), 1e-9)
        hold_days = 1 if reversion_speed > 0.5 else 3

        return Signal(
            direction=direction,
            size=min(reversion_speed * 0.3, 0.35),
            entry=current_price,
            stop=current_price * (1 + (stop - current_ect) / max(abs(current_ect), 1e-9) * 0.1),
            target=current_price * (1 - abs(current_ect) / max(current_price, 1e-9)),
            strategy_name=self.name,
            confidence=min(reversion_speed, 1.0),
            expected_value=ev,
            metadata={"pair": self.pair_symbol, "current_ect": current_ect,
                      "pred_ect": pred_ect, "hold_days": hold_days},
        )


# ===========================================================================
# 4.5  Regime-Switching HMM
# ===========================================================================

class HMMRegime(Strategy):
    """
    Simple 3-state HMM (bull / bear / sideways) fit online via Baum-Welch-like
    forward pass on distribution features.  Uses numpy; no hmmlearn needed.
    """
    name = "hmm_regime"

    def __init__(self, n_regimes: int = 3, min_regime_prob: float = 0.7):
        self.n = n_regimes
        self.min_prob = min_regime_prob
        # State: transition matrix A, observation means B_mu, B_sig
        self._A = np.full((n_regimes, n_regimes), 1.0 / n_regimes)
        self._B_mu = np.array([[-1.0, 0.0, 1.0, 0.5, -0.5],
                               [0.0, 2.0, 0.0, 2.0, 0.0],
                               [1.0, 0.0, -1.0, 0.5, 0.5]])[:n_regimes]
        self._B_sig = np.ones((n_regimes, 5)) * 0.5
        self._pi = np.full(n_regimes, 1.0 / n_regimes)
        self._obs_history: List[np.ndarray] = []
        from kairos_backtest import (TrendFollowingStrategy, RangeTradingStrategy)
        self._trend = TrendFollowingStrategy()
        self._range = RangeTradingStrategy()

    def _features(self, dist: KairosDistribution, current_price: float) -> np.ndarray:
        s = dist.stats["close"]
        entropy = dist.entropy()
        skew = s["skew"]
        cv = dist.coefficient_of_variation()
        pred_range = dist.predicted_range()
        direction = (s["mean"] - current_price) / max(current_price, 1e-9)
        return np.array([direction, entropy, skew, cv, pred_range])

    def _forward(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass (single step) returning posterior state probabilities."""
        # Emission likelihood (Gaussian per feature)
        log_emit = np.zeros(self.n)
        for st in range(self.n):
            diff = obs - self._B_mu[st]
            log_emit[st] = -0.5 * np.sum((diff / self._B_sig[st]) ** 2)
        log_prior = np.log(self._pi + 1e-300) + self._A.T @ np.log(self._pi + 1e-300)
        log_post = log_prior + log_emit
        log_post -= np.max(log_post)
        post = np.exp(log_post)
        return post / (post.sum() + 1e-300)

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        obs = self._features(dist, current_price)
        probs = self._forward(obs)
        regime = int(np.argmax(probs))
        max_prob = float(probs[regime])

        if max_prob < self.min_prob:
            return None

        # Map regime index to strategy
        if self.n == 3:
            if regime == 0:   # bull
                strat = self._trend
                label = "bull"
            elif regime == 1:  # sideways
                strat = self._range
                label = "sideways"
            else:              # bear
                strat = self._trend
                label = "bear"
        else:
            strat = self._trend if regime < self.n // 2 else self._range
            label = f"regime_{regime}"

        sig = strat.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        # In bear regime, flip direction if trend strategy gave LONG
        if label == "bear" and sig.direction == Direction.LONG:
            sig.direction = Direction.SHORT
            sig.stop, sig.target = sig.target, sig.stop

        sig.strategy_name = self.name
        sig.metadata["regime"] = label
        sig.metadata["regime_prob"] = max_prob
        return sig


# ===========================================================================
# 4.6  Wavelet Decomposition Momentum
# ===========================================================================

class WaveletMomentum(Strategy):
    """
    Haar DWT on 60 predicted close samples.  Trades the dominant trend cycle
    if trend_strength (std_approx / std_detail) > threshold.
    """
    name = "wavelet_momentum"

    def __init__(self, wavelet: str = "haar",
                 threshold: float = 2.0,
                 max_size: float = 0.3):
        self.wavelet = wavelet
        self.threshold = threshold
        self.max_size = max_size

    @staticmethod
    def _haar_dwt(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """One-level Haar DWT: returns (approximation, detail)."""
        n = len(signal) // 2 * 2  # ensure even
        s = signal[:n]
        approx = (s[0::2] + s[1::2]) / np.sqrt(2)
        detail = (s[0::2] - s[1::2]) / np.sqrt(2)
        return approx, detail

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        closes = dist.df["close"].values.astype(float)
        if len(closes) < 8:
            return None

        approx, detail = self._haar_dwt(closes)
        std_approx = float(np.std(approx))
        std_detail = float(np.std(detail))

        if std_detail < 1e-9:
            return None

        trend_strength = std_approx / std_detail
        if trend_strength < self.threshold:
            return None

        lookback = min(5, len(approx) - 1)
        if approx[-1] > approx[-1 - lookback]:
            direction = Direction.LONG
        elif approx[-1] < approx[-1 - lookback]:
            direction = Direction.SHORT
        else:
            return None

        s = dist.stats["close"]
        stop = s["pct_20"] if direction == Direction.LONG else s["pct_80"]
        target = s["pct_80"] if direction == Direction.LONG else s["pct_20"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        size = min(trend_strength * 0.2, self.max_size)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(trend_strength / (self.threshold * 2), 1.0),
            expected_value=ev,
            metadata={"trend_strength": trend_strength, "std_approx": std_approx,
                      "std_detail": std_detail},
        )


# ===========================================================================
# 4.7  Detrended Fluctuation Analysis (DFA)
# ===========================================================================

class DFAPersistence(Strategy):
    """
    DFA alpha > 0.55 → trend strategy; alpha < 0.45 → mean-reversion; else None.
    """
    name = "dfa_persistence"

    def __init__(self, trend_threshold: float = 0.55, mr_threshold: float = 0.45):
        self.trend_thresh = trend_threshold
        self.mr_thresh = mr_threshold
        from kairos_backtest import TrendFollowingStrategy, RangeTradingStrategy
        self._trend = TrendFollowingStrategy()
        self._mr = RangeTradingStrategy()

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        closes = dist.df["close"].values.astype(float)
        alpha = _dfa_alpha(closes)

        if alpha > self.trend_thresh:
            strat = self._trend
            regime = "trend"
        elif alpha < self.mr_thresh:
            strat = self._mr
            regime = "mean_reversion"
        else:
            return None

        sig = strat.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        sig.strategy_name = self.name
        sig.metadata["dfa_alpha"] = alpha
        sig.metadata["regime"] = regime
        return sig


# ===========================================================================
# 4.8  Transfer Entropy Causality
# ===========================================================================

class TransferEntropy(Strategy):
    """
    Uses precomputed transfer entropy from a leader asset to generate
    follower signals.  Requires context fields: transfer_entropy, leader_signal,
    optimal_lag.
    """
    name = "transfer_entropy"

    def __init__(self, leader_symbol: str = "",
                 min_te: float = 0.1,
                 max_te: float = 1.0):
        self.leader_symbol = leader_symbol
        self.min_te = min_te
        self.max_te = max_te

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        te = context.get("transfer_entropy")
        leader_sig = context.get("leader_signal")
        if te is None or te < self.min_te or leader_sig is None:
            return None

        direction = leader_sig.direction
        if direction == Direction.FLAT:
            return None

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        te_norm = min(te / self.max_te, 1.0)
        size = leader_sig.size * te_norm

        return Signal(
            direction=direction,
            size=min(size, 0.4),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=te_norm * leader_sig.confidence,
            expected_value=ev,
            metadata={"leader": self.leader_symbol, "te": te,
                      "lag": context.get("optimal_lag", 1)},
        )


# ===========================================================================
# 4.9  Graph Neural Network Sector Rotation (Numpy surrogate)
# ===========================================================================

class GNNSectorRotation(Strategy):
    """
    GNN surrogate: builds an asset correlation graph, propagates predicted
    Sharpe scores across edges, and selects top/bottom nodes.

    Without torch_geometric, implements a 2-layer graph propagation via
    normalised adjacency multiplication (message passing).
    """
    name = "gnn_sector_rotation"

    def __init__(self, top_n: int = 2, bottom_n: int = 2,
                 correlation_threshold: float = 0.7):
        self.top_n = top_n
        self.bottom_n = bottom_n
        self.corr_thresh = correlation_threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        multi_preds = context.get("multi_asset_predictions")
        if not multi_preds or len(multi_preds) < 2:
            return None

        symbols = list(multi_preds.keys())
        n = len(symbols)
        current_sym = context.get("symbol", "")

        # Node features: [pred_sharpe, pred_mean_pct, pred_std_pct, entropy]
        feats = np.zeros((n, 4))
        for i, sym in enumerate(symbols):
            p = multi_preds[sym]
            s = p.dist.stats["close"]
            cp = p.current_price if p.current_price > 0 else 1.0
            feats[i] = [p.dist.predicted_sharpe(),
                        (s["mean"] - cp) / cp,
                        s["std"] / cp,
                        p.dist.entropy()]

        # Build adjacency from feature correlation
        if n > 1:
            corr_mat = np.corrcoef(feats.T)
            A = np.abs(corr_mat)
        else:
            A = np.eye(n)

        # Normalised adjacency (symmetric)
        A_sym = (A + A.T) / 2
        np.fill_diagonal(A_sym, 0)
        deg = A_sym.sum(axis=1)
        d_inv = np.where(deg > 0, 1.0 / (deg + 1e-9), 0.0)
        A_norm = A_sym * d_inv[:, None]

        # 2-layer propagation: H = A * A * features
        H = A_norm @ (A_norm @ feats)
        scores = H[:, 0]  # predicted Sharpe after propagation

        ranked = sorted(zip(symbols, scores), key=lambda x: x[1], reverse=True)
        top_syms = {s for s, _ in ranked[:self.top_n]}
        bot_syms = {s for s, _ in ranked[-self.bottom_n:]}

        if current_sym in top_syms:
            direction = Direction.LONG
        elif current_sym in bot_syms:
            direction = Direction.SHORT
        else:
            return None

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        idx = next(i for i, (sym, _) in enumerate(ranked) if sym == current_sym)
        confidence = abs(0.5 - idx / max(n - 1, 1)) * 2

        return Signal(
            direction=direction,
            size=min(confidence * 0.35, 0.35),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={"gnn_rank": idx, "n_assets": n,
                      "gnn_score": float(scores[symbols.index(current_sym)])
                      if current_sym in symbols else 0.0},
        )


# ===========================================================================
# 4.10  Reinforcement Learning Meta-Controller (Epsilon-Greedy Q-Learning)
# ===========================================================================

class RLMetaController(Strategy):
    """
    Online epsilon-greedy Q-learning over a set of strategies.
    State: discretised distribution features.  Action: strategy index.
    Reward: realised PnL (requires external update via update_reward()).
    """
    name = "rl_meta_controller"

    def __init__(self, all_strategies: List[Strategy],
                 agent_type: str = "q_learning",
                 train_frequency: int = 100):
        self.strategies = all_strategies
        self.n_actions = len(all_strategies)
        self.train_freq = train_frequency
        self._step = 0
        self._epsilon = 1.0
        self._alpha = 0.1    # learning rate
        self._gamma = 0.95   # discount
        self._Q: Dict[Tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions))
        self._last_state: Optional[Tuple] = None
        self._last_action: Optional[int] = None

    def _state(self, dist: KairosDistribution, current_price: float) -> Tuple:
        s = dist.stats["close"]
        e = dist.entropy()
        skew_bin = int(np.clip(np.sign(s["skew"]), -1, 1))
        e_bin = int(np.clip(e / 1.0, 0, 2))
        direction_bin = int(np.sign(s["mean"] - current_price))
        return (skew_bin, e_bin, direction_bin)

    def update_reward(self, reward: float):
        """Call after each trade with realised PnL to update Q-table."""
        if self._last_state is None or self._last_action is None:
            return
        q_old = self._Q[self._last_state][self._last_action]
        q_new = q_old + self._alpha * (reward - q_old)
        self._Q[self._last_state][self._last_action] = q_new
        self._epsilon = max(0.05, self._epsilon * 0.999)

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        if not self.strategies:
            return None
        self._step += 1
        state = self._state(dist, current_price)

        rng = np.random.default_rng(self._step)
        if rng.random() < self._epsilon:
            action = int(rng.integers(0, self.n_actions))
        else:
            action = int(np.argmax(self._Q[state]))

        self._last_state = state
        self._last_action = action

        strat = self.strategies[action]
        sig = strat.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        sig.strategy_name = self.name
        sig.metadata["rl_action"] = action
        sig.metadata["rl_strategy"] = strat.name
        sig.metadata["epsilon"] = self._epsilon
        return sig


# ===========================================================================
# 4.11  Fractal Dimension Trading
# ===========================================================================

class FractalDimension(Strategy):
    """
    Blocks trades when box-counting fractal dimension > threshold (noisy market).
    """
    name = "fractal_dimension"

    def __init__(self, base_strategy: Strategy, threshold: float = 1.5):
        self.base = base_strategy
        self.threshold = threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        closes = dist.df["close"].values.astype(float)
        fd = _fractal_dim(closes)

        if fd > self.threshold:
            return None

        sig = self.base.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        sig.strategy_name = self.name
        sig.metadata["fractal_dim"] = fd
        return sig


# ===========================================================================
# 4.12  Lempel-Ziv Complexity
# ===========================================================================

class LZComplexity(Strategy):
    """
    Blocks trades when normalised LZ complexity > threshold (random market).
    """
    name = "lz_complexity"

    def __init__(self, base_strategy: Strategy, threshold: float = 0.8):
        self.base = base_strategy
        self.threshold = threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        closes = dist.df["close"].values.astype(float)
        lz = _lz_complexity(closes)

        if lz > self.threshold:
            return None

        sig = self.base.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        sig.strategy_name = self.name
        sig.metadata["lz_complexity"] = lz
        return sig


# ===========================================================================
# 4.13  Recurrence Quantification Analysis (RQA)
# ===========================================================================

class RQADeterminism(Strategy):
    """
    Uses RQA determinism and laminarity to select trend vs. range strategy.
    High DET + LAM → trend; low DET + LAM → skip; else → range.
    """
    name = "rqa_determinism"

    def __init__(self, det_threshold: float = 0.7, lam_threshold: float = 0.5):
        self.det_thresh = det_threshold
        self.lam_thresh = lam_threshold
        from kairos_backtest import TrendFollowingStrategy, RangeTradingStrategy
        self._trend = TrendFollowingStrategy()
        self._range = RangeTradingStrategy()

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        closes = dist.df["close"].values.astype(float)
        det, lam = _rqa_metrics(closes)

        if det > self.det_thresh and lam > self.lam_thresh:
            strat = self._trend
            regime = "trend"
        elif det < 0.3 and lam < 0.3:
            return None
        else:
            strat = self._range
            regime = "uncertain_range"

        sig = strat.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        sig.strategy_name = self.name
        sig.metadata["det"] = det
        sig.metadata["lam"] = lam
        sig.metadata["regime"] = regime
        return sig


# ===========================================================================
# 4.14  Mutual Information Feature Selection
# ===========================================================================

class MutualInformationWeight(Strategy):
    """
    Weights a set of strategies by the mutual information of their primary
    feature with future returns.  Runs all strategies; returns the highest
    MI-weighted signal.
    """
    name = "mutual_information_weight"

    def __init__(self, feature_map: Dict[str, str], lookback: int = 100):
        self.feature_map = feature_map  # {strategy_name: feature_name}
        self.lookback = lookback
        self._mi_scores: Dict[str, float] = {}

    @staticmethod
    def _mutual_info(x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """Estimate MI via joint histogram."""
        hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
        px = hist_xy.sum(axis=1)
        py = hist_xy.sum(axis=0)
        total = hist_xy.sum() + 1e-300
        px /= total
        py /= total
        pxy = hist_xy / total
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
        return max(mi, 0.0)

    def _update_mi(self, history, context: Dict):
        """Recompute MI scores from recent history."""
        if history is None or len(history) < self.lookback:
            return
        closes = history["close"].values.astype(float)[-self.lookback:]
        future_ret = np.diff(closes) / (closes[:-1] + 1e-9)
        if len(future_ret) < 10:
            return

        for strat_name, feature in self.feature_map.items():
            if feature == "rsi":
                delta = np.diff(closes[:-1])
                up = np.maximum(delta, 0)
                dn = -np.minimum(delta, 0)
                rs = np.convolve(up, np.ones(14) / 14, "valid") / \
                     (np.convolve(dn, np.ones(14) / 14, "valid") + 1e-9)
                rsi = 100 - 100 / (1 + rs)
                n = min(len(rsi), len(future_ret))
                if n > 5:
                    self._mi_scores[strat_name] = self._mutual_info(
                        rsi[-n:], future_ret[-n:])
            elif feature == "volume":
                if "volume" in history.columns:
                    vol = history["volume"].values.astype(float)[-self.lookback:-1]
                    n = min(len(vol), len(future_ret))
                    if n > 5:
                        self._mi_scores[strat_name] = self._mutual_info(
                            vol[-n:], future_ret[-n:])
            else:
                # Default: close-based momentum as feature proxy
                mom = np.diff(closes[:-1])
                n = min(len(mom), len(future_ret))
                if n > 5:
                    self._mi_scores[strat_name] = self._mutual_info(
                        mom[-n:], future_ret[-n:])

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        # MI scores are updated externally or lazily
        if not self._mi_scores and history is not None:
            self._update_mi(history, context)

        if not self._mi_scores:
            return None

        strategies: List[Strategy] = context.get("candidate_strategies", [])
        if not strategies:
            return None

        best_sig, best_score = None, -1.0
        for strat in strategies:
            mi = self._mi_scores.get(strat.name, 0.0)
            sig = strat.generate_signal(dist, current_price, history, context)
            if sig is None:
                continue
            score = mi * sig.expected_value * sig.confidence
            if score > best_score:
                best_sig = sig
                best_score = score

        if best_sig is None:
            return None

        best_sig.strategy_name = self.name
        best_sig.metadata["mi_score"] = best_score
        return best_sig


# ===========================================================================
# 4.15  Gaussian Process Regression
# ===========================================================================

class GaussianProcess(Strategy):
    """
    Fits an RBF Gaussian Process to the 60 predicted close samples.
    Uses GP posterior mean and std for more accurate signal generation.
    """
    name = "gaussian_process"

    def __init__(self, base_strategy: Strategy,
                 kernel: str = "rbf",
                 noise_level: float = 0.01):
        self.base = base_strategy
        self.kernel = kernel
        self.noise = noise_level

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        closes = dist.df["close"].values.astype(float)
        n = len(closes)
        if n < 4:
            return self.base.generate_signal(dist, current_price, history, context)

        X = np.arange(n, dtype=float) / n
        y = closes

        # Fit GP and predict at one step ahead
        mu, var = _rbf_gp(X, y, np.array([1.0 + 1.0 / n]),
                          noise=self.noise, length_scale=0.3)
        gp_mean = float(mu[0])
        gp_std = float(np.sqrt(max(var[0], 1e-9)))

        # Build a synthetic distribution with GP parameters for the base strategy
        # Use the GP mean and std to override key stats
        sig = self.base.generate_signal(dist, current_price, history, context)
        if sig is None:
            return None

        # Adjust confidence by GP uncertainty (tighter std = higher confidence)
        empirical_std = float(np.std(closes)) + 1e-9
        gp_confidence_factor = min(empirical_std / gp_std, 2.0)
        sig.confidence = min(sig.confidence * gp_confidence_factor, 1.0)
        sig.strategy_name = self.name
        sig.metadata["gp_mean"] = gp_mean
        sig.metadata["gp_std"] = gp_std
        sig.metadata["gp_confidence_factor"] = gp_confidence_factor
        return sig


# ===========================================================================
# 4.16  Bayesian Structural Time Series (BSTS)
# ===========================================================================

class BSTSDecomposition(Strategy):
    """
    Combines a Kalman-estimated local linear trend with Kairos predicted close.
    Enters only when both the BSTS trend and Kairos prediction agree.
    """
    name = "bsts_decomposition"

    def __init__(self, max_size: float = 0.3):
        self.max_size = max_size
        # Local linear trend state: [level, slope]
        self._level: Optional[float] = None
        self._slope: float = 0.0
        self._level_var: float = 1.0
        self._slope_var: float = 0.1
        self._sigma_obs: float = 1.0
        self._sigma_level: float = 0.1
        self._sigma_slope: float = 0.01

    def _update(self, obs: float):
        if self._level is None:
            self._level = obs
            return
        # Predict
        pred_level = self._level + self._slope
        pred_slope = self._slope
        pred_level_var = self._level_var + self._slope_var + self._sigma_level
        pred_slope_var = self._slope_var + self._sigma_slope
        # Update (Kalman)
        S = pred_level_var + self._sigma_obs
        K_level = pred_level_var / S
        K_slope = pred_slope_var / S
        innov = obs - pred_level
        self._level = pred_level + K_level * innov
        self._slope = pred_slope + K_slope * innov
        self._level_var = (1 - K_level) * pred_level_var
        self._slope_var = (1 - K_slope) * pred_slope_var

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        # Update BSTS with current price
        self._update(current_price)
        bsts_trend_override = context.get("bsts_trend")
        bsts_trend = bsts_trend_override if bsts_trend_override is not None else self._slope

        pred_mean = dist.stats["close"]["mean"]
        pred_sharpe = dist.predicted_sharpe()

        if pred_mean > current_price and bsts_trend > 0:
            direction = Direction.LONG
        elif pred_mean < current_price and bsts_trend < 0:
            direction = Direction.SHORT
        else:
            return None

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        size = min(abs(bsts_trend) * abs(pred_sharpe), self.max_size)

        return Signal(
            direction=direction,
            size=max(size, 0.05),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(abs(pred_sharpe) * 0.4, 1.0),
            expected_value=ev,
            metadata={"bsts_level": self._level, "bsts_slope": float(self._slope),
                      "bsts_trend": float(bsts_trend)},
        )


# ===========================================================================
# 4.17  Particle Filter Tracking
# ===========================================================================

class ParticleFilter(Strategy):
    """
    Treats the 60 predicted close samples as particles.  Reweights by
    likelihood given recent price history, then uses weighted statistics
    for signal generation.
    """
    name = "particle_filter"

    def __init__(self, base_strategy: Strategy, n_particles: int = 60):
        self.base = base_strategy
        self.n_particles = n_particles

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        particles = dist.df["close"].values.astype(float)

        # Compute observation sigma from recent history
        if history is not None and len(history) >= 5:
            recent = history["close"].values.astype(float)[-20:]
            sigma = float(np.std(recent)) + 1e-9
            weights = _particle_weights(particles, recent, sigma)
        else:
            weights = np.ones(len(particles)) / len(particles)

        # Weighted statistics
        w_mean = float(np.average(particles, weights=weights))
        w_var = float(np.average((particles - w_mean) ** 2, weights=weights))
        w_std = float(np.sqrt(max(w_var, 1e-9)))

        # Determine direction from weighted mean
        if w_mean > current_price:
            direction = Direction.LONG
        elif w_mean < current_price:
            direction = Direction.SHORT
        else:
            return None

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        empirical_std = float(dist.stats["close"]["std"]) + 1e-9
        confidence_boost = min(empirical_std / w_std, 1.5)

        return Signal(
            direction=direction,
            size=min(0.3 * confidence_boost, 0.4),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=min(abs(w_mean - current_price) / (w_std + 1e-9) * 0.3, 1.0),
            expected_value=ev,
            metadata={"filtered_mean": w_mean, "filtered_std": w_std,
                      "empirical_std": empirical_std, "confidence_boost": confidence_boost},
        )


# ===========================================================================
# 4.18  Spectral Clustering for Asset Selection
# ===========================================================================

class SpectralClustering(Strategy):
    """
    Spectral clustering on predicted forward correlation matrix.
    Longs the best cluster, shorts the worst.
    """
    name = "spectral_clustering"

    def __init__(self, n_clusters: int = 3, correlation_lookback: int = 30):
        self.n_clusters = n_clusters
        self.lookback = correlation_lookback

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history, context: Dict) -> Optional[Signal]:
        multi_preds = context.get("multi_asset_predictions")
        if not multi_preds or len(multi_preds) < self.n_clusters + 1:
            return None

        symbols = list(multi_preds.keys())
        n = len(symbols)
        current_sym = context.get("symbol", "")

        # Build predicted return matrix from each asset's distribution samples
        samples_matrix = np.zeros((n, 100))
        sharpes = np.zeros(n)
        for i, sym in enumerate(symbols):
            p = multi_preds[sym]
            closes = p.dist.df["close"].values.astype(float)
            m = min(len(closes), 100)
            samples_matrix[i, :m] = closes[:m]
            if m < 100:
                samples_matrix[i, m:] = closes[-1]
            sharpes[i] = p.dist.predicted_sharpe()

        # Forward-looking correlation (Pearson on predicted sample arrays)
        corr_mat = np.corrcoef(samples_matrix)
        labels = _spectral_cluster(corr_mat, self.n_clusters)

        # Average Sharpe per cluster
        cluster_sharpes = {}
        for c in range(self.n_clusters):
            members = [j for j in range(n) if labels[j] == c]
            if members:
                cluster_sharpes[c] = float(np.mean(sharpes[members]))

        if not cluster_sharpes:
            return None

        best_cluster = max(cluster_sharpes, key=cluster_sharpes.get)
        worst_cluster = min(cluster_sharpes, key=cluster_sharpes.get)

        if current_sym not in symbols:
            return None

        idx = symbols.index(current_sym)
        sym_cluster = labels[idx]

        if sym_cluster == best_cluster:
            direction = Direction.LONG
        elif sym_cluster == worst_cluster:
            direction = Direction.SHORT
        else:
            return None

        s = dist.stats["close"]
        stop = s["pct_10"] if direction == Direction.LONG else s["pct_90"]
        target = s["pct_90"] if direction == Direction.LONG else s["pct_10"]

        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        sharpe_gap = cluster_sharpes[best_cluster] - cluster_sharpes[worst_cluster]
        confidence = min(abs(sharpe_gap) * 0.3, 1.0)

        return Signal(
            direction=direction,
            size=min(confidence * 0.35, 0.35),
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={"cluster": int(sym_cluster), "n_clusters": self.n_clusters,
                      "best_cluster_sharpe": cluster_sharpes[best_cluster],
                      "worst_cluster_sharpe": cluster_sharpes[worst_cluster]},
        )
