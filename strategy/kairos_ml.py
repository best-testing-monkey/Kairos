"""
kairos_ml.py
============
Machine Learning strategies and filter wrappers for Kairos.

Module: Meta-Labeling (López de Prado)
- Wrapper around any base strategy that labels historical signals by outcome
  (profit-take/stop/time via signal bracket), trains a secondary classifier
  P(signal wins) on features {entropy, kurtosis, skew, CDF position, ATR ratio,
  trailing strategy hit-rate, regime id}, and sizes live signals by predicted
  probability (vetoing below p_min).
- Classifier: logistic regression (numpy IRLS, no sklearn).
- Warm-up: pass-through for first 60 labeled signals.
"""

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

from kairos_backtest import Strategy, Signal, Direction


# =============================================================================
# Meta-Labeling Classifier
# =============================================================================

class LogisticRegressionIRLS:
    """Logistic regression via Iteratively Reweighted Least Squares (IRLS).

    Fits P(y=1|X) = sigmoid(X @ w + b) by maximizing log-likelihood with
    L2 ridge regularization (alpha=1e-3 default).
    """

    def __init__(self, alpha: float = 1e-3, max_iter: int = 50, tol: float = 1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit via IRLS: Newton-Raphson on log-likelihood with L2 penalty.

        Args:
            X: shape (n_samples, n_features)
            y: shape (n_samples,) with values 0 or 1
        """
        n_samples, n_features = X.shape

        # Augmented design matrix with a bias column (bias is not penalized
        # separately; the small ridge on it keeps the Hessian well-conditioned).
        Xa = np.hstack([X, np.ones((n_samples, 1))])
        beta = np.zeros(n_features + 1)

        # IRLS iterations (damped Newton-Raphson)
        for _ in range(self.max_iter):
            z = Xa @ beta
            p = self._sigmoid(z)

            residual = y - p
            weights = np.clip(p * (1 - p), 1e-6, 1.0)

            # Gradient with L2 penalty: Xa^T (y - p) - alpha * beta
            grad = Xa.T @ residual - self.alpha * beta

            # Hessian: Xa^T W Xa + alpha I (W applied row-wise, no dense diag)
            H = (Xa * weights[:, None]).T @ Xa + self.alpha * np.eye(n_features + 1)

            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H) @ grad

            # Damp the step on (near-)separable data to prevent oscillation
            step_norm = np.linalg.norm(delta)
            if step_norm > 10.0:
                delta *= 10.0 / step_norm

            beta = beta + delta

            if np.linalg.norm(delta) < self.tol:
                break

        self.w = beta[:-1]
        self.b = float(beta[-1])
        self.fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(y=1|X).

        Args:
            X: shape (n_samples, n_features)

        Returns:
            Predicted probabilities, shape (n_samples,)
        """
        if self.w is None:
            raise ValueError("Model not fitted yet")

        z = X @ self.w + self.b
        return self._sigmoid(z)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1.0 / (1.0 + np.exp(-z))


# =============================================================================
# Meta-Labeling Strategy
# =============================================================================

class MetaLabelStrategy(Strategy):
    """López de Prado meta-labeling wrapper.

    Wraps a base strategy and labels each signal by triple-barrier outcome
    (profit-take/stop/time). Trains a logistic regression classifier on
    features and sizes signals by predicted P(win).

    Warm-up (first 60 labeled signals): pass-through to base strategy.
    After warm-up: filter and size by classifier.

    Parameters:
        base_strategy: Strategy to wrap
        p_min: Minimum predicted probability to accept signal (default 0.55)
        warmup: Number of labeled signals before classifier activates (default 60)
    """

    name = "meta_label"

    def __init__(self, base_strategy: Strategy, p_min: float = 0.55,
                 warmup: int = 60):
        self.base_strategy = base_strategy
        self.p_min = p_min
        self.warmup = warmup

        # Labeled training data: list of (features, label)
        self.labeled_pairs = []

        # Pending signal info for labeling: (features, signal)
        self.pending_signal = None
        self.pending_features = None

        # Classifier and refit counter
        self.classifier = LogisticRegressionIRLS(alpha=1e-3)
        self.fit_count = 0  # Track number of times we've fit
        self.labels_since_last_fit = 0
        self.refit_cadence = 20  # Refit every 20 new labels

        # Trailing hit-rate tracking
        self.trailing_outcomes = deque(maxlen=20)

    def reset(self) -> None:
        """Reset state for walk-forward folds."""
        self.labeled_pairs = []
        self.pending_signal = None
        self.pending_features = None
        self.classifier = LogisticRegressionIRLS(alpha=1e-3)
        self.fit_count = 0
        self.labels_since_last_fit = 0
        self.trailing_outcomes = deque(maxlen=20)

    def generate_signal(self, dist, current_price: float, history: pd.DataFrame,
                        context: Dict[str, Any], **kwargs) -> Optional[Signal]:
        """Generate signal via base strategy, potentially vetoed/sized by classifier.

        Args:
            dist: KairosDistribution for feature extraction
            current_price: Current price (entry point)
            history: Historical OHLCV data
            context: Context dict (may include regime_id)

        Returns:
            Signal or None
        """
        # Get base signal
        base_sig = self.base_strategy.generate_signal(dist, current_price,
                                                      history, context, **kwargs)
        if base_sig is None:
            return None

        # Extract features from this bar
        features = self._extract_features(dist, current_price, history, context)

        # Store pending signal for later labeling
        self.pending_signal = base_sig
        self.pending_features = features

        # During warm-up: pass through unchanged
        if len(self.labeled_pairs) < self.warmup:
            return base_sig

        # After warm-up: lazily fit if we crossed warm-up between refit cadences
        if not self.classifier.fitted:
            self._refit_classifier()
            self.labels_since_last_fit = 0
            if not self.classifier.fitted:
                # Degenerate labeled set (< 2 samples) — pass through
                return base_sig

        features_array = np.array([features])
        p_win = float(self.classifier.predict_proba(features_array)[0])

        # Veto if below threshold
        if p_win < self.p_min:
            return None

        # Scale signal size by predicted probability
        sig_scaled = Signal(
            direction=base_sig.direction,
            size=base_sig.size * p_win,
            entry=base_sig.entry,
            stop=base_sig.stop,
            target=base_sig.target,
            strategy_name=base_sig.strategy_name,
            confidence=base_sig.confidence * p_win,
            expected_value=base_sig.expected_value,
            metadata={**base_sig.metadata, "p_win": p_win}
        )
        return sig_scaled

    def label_last(self, outcome: float) -> None:
        """Label the last pending signal with outcome.

        Args:
            outcome: 1.0 for win, 0.0 for loss
        """
        if self.pending_features is None or self.pending_signal is None:
            return

        # Store labeled pair
        self.labeled_pairs.append((self.pending_features, outcome))

        # Track trailing hit-rate
        self.trailing_outcomes.append(outcome)

        # Update refit counter
        self.labels_since_last_fit += 1

        # Refit if we've accumulated enough new labels
        if self.labels_since_last_fit >= self.refit_cadence:
            self._refit_classifier()
            self.labels_since_last_fit = 0

        # Clear pending
        self.pending_signal = None
        self.pending_features = None

    def _refit_classifier(self) -> None:
        """Refit logistic regression on all labeled pairs."""
        if len(self.labeled_pairs) < 2:
            return  # Need at least 2 samples

        # Convert to arrays
        X = np.array([pair[0] for pair in self.labeled_pairs])
        y = np.array([pair[1] for pair in self.labeled_pairs])

        # Fit classifier
        self.classifier.fit(X, y)
        self.fit_count += 1

    def _extract_features(self, dist, current_price: float, history: pd.DataFrame,
                          context: Dict[str, Any]) -> list:
        """Extract feature vector for classifier.

        Features: [entropy, kurtosis, skew, CDF position, ATR ratio,
                   trailing hit-rate, regime id]

        Args:
            dist: KairosDistribution
            current_price: Current price
            history: Historical data
            context: Context dict

        Returns:
            Feature list (will be converted to numpy array by caller)
        """
        # 1. Entropy (Shannon, range 0 to ln(20) ≈ 3.0)
        entropy = dist.entropy()

        # 2. Kurtosis (excess kurtosis from predicted close distribution)
        stats_close = dist.stats.get("close", {})
        kurtosis = stats_close.get("kurt", 0.0)

        # 3. Skew
        skew = stats_close.get("skew", 0.0)

        # 4. CDF position (P(price < current_price))
        cdf_pos = dist.cdf(current_price)

        # 5. ATR ratio (ATR / current price)
        atr_ratio = MetaLabelStrategy._compute_atr_ratio(history, current_price)

        # 6. Trailing hit-rate over recent labeled outcomes (0.5 if none yet)
        if len(self.trailing_outcomes) > 0:
            trailing_hit_rate = float(np.mean(self.trailing_outcomes))
        else:
            trailing_hit_rate = 0.5

        # 7. Regime ID (default to 0 if not in context)
        regime_id = float(context.get("regime_id", 0))

        return [entropy, kurtosis, skew, cdf_pos, atr_ratio,
                trailing_hit_rate, regime_id]

    @staticmethod
    def _compute_atr_ratio(history: pd.DataFrame, current_price: float,
                           period: int = 14) -> float:
        """Compute ATR ratio: ATR / current_price.

        ATR (Average True Range) measures volatility.

        Args:
            history: Historical OHLCV data (must have 'high', 'low', 'close')
            current_price: Current price (for denominator)
            period: ATR period (default 14)

        Returns:
            ATR / current_price ratio
        """
        if len(history) < period:
            return 0.01  # Default small ratio if insufficient data

        # Compute True Range
        high = history["high"].values
        low = history["low"].values
        close = history["close"].values

        tr_values = []
        for i in range(1, len(close)):
            h_l = high[i] - low[i]
            h_c = abs(high[i] - close[i - 1])
            l_c = abs(low[i] - close[i - 1])
            tr = max(h_l, h_c, l_c)
            tr_values.append(tr)

        if len(tr_values) < period:
            return 0.01

        # ATR is SMA of TR
        atr = np.mean(tr_values[-period:])

        # Return ratio
        if current_price > 0:
            return float(atr / current_price)
        return 0.01
