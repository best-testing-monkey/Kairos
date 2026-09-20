"""
kairos_ml.py
============
Machine Learning strategies and filter wrappers for Kairos.

Module 1: Meta-Labeling (López de Prado)
- Wrapper around any base strategy that labels historical signals by outcome
  (profit-take/stop/time via signal bracket), trains a secondary classifier
  P(signal wins) on features {entropy, kurtosis, skew, CDF position, ATR ratio,
  trailing strategy hit-rate, regime id}, and sizes live signals by predicted
  probability (vetoing below p_min).
- Classifier: logistic regression (numpy IRLS, no sklearn).
- Warm-up: pass-through for first 60 labeled signals.

Module 2: GBM Direction Classifier
- Standalone gradient-boosted tree strategy (50 trees, depth 2, lr 0.1, logloss).
- Features (~15): returns 1/5/20d, RSI(14), ATR/price, volume z-score, day-of-week,
  rolling vol, SMA ratios, Kronos distribution stats.
- Labels: next-day direction from historical closes.
- Trades when P(direction) > p_min AND Kronos agrees.
- Retrain weekly on trailing lookback rows (never per-bar).

Module 3: LPPLS Bubble Detection
- Fits log-periodic power law singularity (Sornette) model via multi-start
  Nelder-Mead nonlinear optimization on 3 params (tc, m, ω), with linear
  params profiled out via least squares.
- Detects bubbles: 0.1<m<0.9, 6<ω<13, tc within (T, T+60], B<0 (super-exponential).
- LPPLSGuardStrategy wrapper: vetoes LONG entries during bubbles, boosts SHORT.

Module 4: ML Bracket Strategy (E18-S04)
- Wrapper that selects optimal stop/target from trained per-candidate
  GradientBoostedStumps classifiers (E18-S03).
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from collections import deque
from typing import Optional, Dict, Any
from scipy.optimize import minimize

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


# =============================================================================
# Gradient Boosted Stumps (Depth-2 Trees)
# =============================================================================

class GradientBoostedStumps:
    """Gradient boosting with depth-2 decision trees (stump-of-stumps).

    Implements binary classification with binary logloss (log-likelihood).
    Each tree is a simple depth-2 tree: root split, then one split per child.
    Fit via stagewise forward descent on residuals.

    Parameters:
        n_trees: Number of weak learners (depth-2 trees) to fit
        lr: Learning rate (shrinkage) applied to each tree's output
        seed: Random seed for reproducibility
    """

    def __init__(self, n_trees: int = 50, lr: float = 0.1, seed: int = 7):
        self.n_trees = n_trees
        self.lr = lr
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.trees = []          # List of fitted trees (dict format)
        self.initial_pred = 0.0  # log(p/(1-p)) of training set
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit gradient boosting model.

        Args:
            X: shape (n_samples, n_features), feature matrix
            y: shape (n_samples,), binary labels {0, 1}
        """
        n_samples = len(y)
        y = np.asarray(y, dtype=float)

        # Initial prediction: log-odds of the training set
        p_init = np.mean(y)
        p_init = np.clip(p_init, 1e-6, 1.0 - 1e-6)
        self.initial_pred = float(np.log(p_init / (1.0 - p_init)))

        # Initialize predictions (on log-odds scale)
        F = np.full(n_samples, self.initial_pred, dtype=float)

        self.trees = []

        # Stagewise fitting
        for _ in range(self.n_trees):
            # Compute pseudo-residuals (gradient of logloss)
            pred_proba = self._sigmoid(F)
            residuals = y - pred_proba

            # Fit a depth-2 tree to residuals
            tree = self._fit_depth2_tree(X, residuals)
            self.trees.append(tree)

            # Update predictions
            tree_pred = self._predict_tree(X, tree)
            F = F + self.lr * tree_pred

        self.fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(y=1|X).

        Args:
            X: shape (n_samples, n_features)

        Returns:
            Predicted probabilities, shape (n_samples,)
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        n_samples = len(X)
        F = np.full(n_samples, self.initial_pred, dtype=float)

        for tree in self.trees:
            tree_pred = self._predict_tree(X, tree)
            F = F + self.lr * tree_pred

        return self._sigmoid(F)

    def _fit_depth2_tree(self, X: np.ndarray, residuals: np.ndarray) -> dict:
        """Fit a single depth-2 tree via exhaustive search.

        Returns a dict representing the tree:
            {
                'feature': int (feature index for root split),
                'threshold': float (threshold for root split),
                'left_leaf': float (value for left child),
                'right_leaf': float (value for right child)
            }

        For simplicity, each leaf is a constant (the mean of residuals in that region).
        """
        n_samples, n_features = X.shape

        best_loss = float('inf')
        best_tree = None

        # Try each feature as root split
        for feat_idx in range(n_features):
            X_feat = X[:, feat_idx]

            # Use quantile-based candidate thresholds
            candidates = np.percentile(X_feat, [10, 25, 50, 75, 90])

            for threshold in candidates:
                # Root split
                left_mask = X_feat <= threshold
                right_mask = ~left_mask

                n_left = left_mask.sum()
                n_right = right_mask.sum()

                # Skip if split is too imbalanced or empty
                if n_left == 0 or n_right == 0:
                    continue

                # For each child, fit a constant (best leaf is mean residual)
                left_residuals = residuals[left_mask]
                right_residuals = residuals[right_mask]

                left_leaf = float(np.mean(left_residuals))
                right_leaf = float(np.mean(right_residuals))

                # Compute squared error loss (for stability)
                left_pred = np.full_like(left_residuals, left_leaf)
                right_pred = np.full_like(right_residuals, right_leaf)
                loss = float(np.mean((left_residuals - left_pred) ** 2)) * n_left
                loss += float(np.mean((right_residuals - right_pred) ** 2)) * n_right
                loss /= n_samples

                if loss < best_loss:
                    best_loss = loss
                    best_tree = {
                        'feature': feat_idx,
                        'threshold': float(threshold),
                        'left_leaf': left_leaf,
                        'right_leaf': right_leaf
                    }

        # Fallback if no valid split found
        if best_tree is None:
            best_tree = {
                'feature': 0,
                'threshold': 0.0,
                'left_leaf': float(np.mean(residuals)),
                'right_leaf': float(np.mean(residuals))
            }

        return best_tree

    def _predict_tree(self, X: np.ndarray, tree: dict) -> np.ndarray:
        """Predict using a single tree."""
        X_feat = X[:, tree['feature']]
        predictions = np.where(X_feat <= tree['threshold'],
                                tree['left_leaf'],
                                tree['right_leaf'])
        return predictions

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))


# =============================================================================
# GBM Direction Strategy
# =============================================================================

class GBMDirectionStrategy(Strategy):
    """Gradient-boosted direction classifier.

    Standalone strategy that predicts next-day direction from technical features.
    Emits LONG when classifier predicts up AND Kronos mean > current price.
    Emits SHORT when classifier predicts down AND Kronos mean < current price.

    Parameters:
        lookback: Trailing window for retraining (default 500)
        retrain_days: Number of signal calls between retrains (default 5)
        p_min: Minimum predicted probability to emit signal (default 0.6)
        seed: Random seed for GBM reproducibility (default 7)
    """

    name = "gbm_direction"

    def __init__(self, lookback: int = 500, retrain_days: int = 5,
                 p_min: float = 0.6, seed: int = 7):
        self.lookback = lookback
        self.retrain_days = retrain_days
        self.p_min = p_min
        self.seed = seed

        # Model state
        self.gbm = GradientBoostedStumps(n_trees=50, lr=0.1, seed=seed)
        self.fit_count = 0
        self.call_count = 0  # Track calls to generate_signal

    def reset(self) -> None:
        """Reset state for walk-forward folds."""
        self.gbm = GradientBoostedStumps(n_trees=50, lr=0.1, seed=self.seed)
        self.fit_count = 0
        self.call_count = 0

    def generate_signal(self, dist, current_price: float, history: pd.DataFrame,
                        context: Dict[str, Any], **kwargs) -> Optional[Signal]:
        """Generate signal based on GBM direction prediction and Kronos agreement.

        Args:
            dist: KairosDistribution for Kronos stats
            current_price: Current price (entry point)
            history: Historical OHLCV data (at least 120 rows for features)
            context: Context dict (unused but matches signature)

        Returns:
            Signal (LONG/SHORT with kelly sizing) or None
        """
        self.call_count += 1

        # Insufficient history for feature extraction
        if len(history) < 120:
            return None

        # Refit every retrain_days calls
        if self.call_count % self.retrain_days == 0:
            self._refit_model(history)

        # If model not fitted yet, return None
        if not self.gbm.fitted:
            return None

        # Extract features for current bar
        features = self._extract_features(history, dist)
        if features is None:
            return None

        X_current = np.array([features])
        p_up = float(self.gbm.predict_proba(X_current)[0])
        p_down = 1.0 - p_up

        # Get Kronos prediction
        kronos_mean = dist.stats.get("close", {}).get("mean", current_price)

        # Decision logic
        if p_up > self.p_min and kronos_mean > current_price:
            # LONG signal
            direction = Direction.LONG
            stop = dist.stats["close"].get("pct_15", current_price * 0.97)
            target = dist.stats["close"].get("pct_85", current_price * 1.03)
        elif p_down > self.p_min and kronos_mean < current_price:
            # SHORT signal (reversed brackets)
            direction = Direction.SHORT
            stop = dist.stats["close"].get("pct_85", current_price * 1.03)
            target = dist.stats["close"].get("pct_15", current_price * 0.97)
        else:
            return None

        # Size: min(kelly * 0.5, 1.0)
        kelly = dist.kelly_fraction(current_price, target, stop)
        size = min(kelly * 0.5, 1.0)

        # Confidence: predicted probability
        confidence = max(p_up, p_down)

        # Expected value
        ev = dist.expected_value(current_price, target, stop)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={"p_direction": max(p_up, p_down), "fit_count": self.fit_count}
        )

    def _refit_model(self, history: pd.DataFrame) -> None:
        """Refit GBM on trailing lookback rows.

        Args:
            history: Full historical data
        """
        # Use trailing lookback rows
        train_data = history.iloc[-self.lookback:].copy()

        if len(train_data) < 120:
            return

        # Extract features for all rows except the last (no next-day label)
        X_list = []
        y_list = []

        for i in range(len(train_data) - 1):
            row_history = train_data.iloc[:i + 1]
            features = self._extract_features(row_history, None)
            if features is None:
                continue

            X_list.append(features)

            # Label: 1 if next close > current close, else 0
            current_close = float(train_data.iloc[i]["close"])
            next_close = float(train_data.iloc[i + 1]["close"])
            label = 1.0 if next_close > current_close else 0.0
            y_list.append(label)

        if len(X_list) < 10:
            # Not enough training data
            return

        X = np.array(X_list)
        y = np.array(y_list)

        # Fit new model
        self.gbm.fit(X, y)
        self.fit_count += 1

    def _extract_features(self, history: pd.DataFrame,
                          dist: Optional = None) -> Optional[list]:
        """Extract ~15 technical features from history and distribution.

        Features:
        1-3: Returns over 1/5/20 days
        4: RSI(14)
        5: ATR(14) / current price
        6: Volume z-score(20)
        7-11: Day-of-week one-hot (5 features)
        12: Rolling volatility(20)
        13: close / SMA(20) - 1
        14: close / SMA(50) - 1
        15: Kronos entropy (if dist provided)

        Args:
            history: Historical data (at least 50 rows)
            dist: Optional KairosDistribution for entropy

        Returns:
            Feature list or None if insufficient data
        """
        if len(history) < 50:
            return None

        close = history["close"].values
        volume = history["volume"].values
        high = history["high"].values
        low = history["low"].values

        features = []

        # 1-3: Returns at 1/5/20 days (log returns)
        for period in [1, 5, 20]:
            if len(close) >= period + 1:
                ret = float(np.log(close[-1] / close[-(period + 1)]))
            else:
                ret = 0.0
            features.append(ret)

        # 4: RSI(14)
        rsi = self._compute_rsi(close, period=14)
        features.append(rsi)

        # 5: ATR(14) / current price
        atr_ratio = self._compute_atr_ratio_gbs(high, low, close)
        features.append(atr_ratio)

        # 6: Volume z-score(20)
        vol_zscore = self._compute_volume_zscore(volume)
        features.append(vol_zscore)

        # 7-11: Day-of-week one-hot (5 features, assuming we have the date)
        dow_features = self._compute_day_of_week_onehot(history)
        features.extend(dow_features)

        # 12: Rolling volatility(20)
        rolling_vol = self._compute_rolling_volatility(close, period=20)
        features.append(rolling_vol)

        # 13: close / SMA(20) - 1
        sma20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        features.append(float(close[-1] / sma20 - 1.0))

        # 14: close / SMA(50) - 1
        sma50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
        features.append(float(close[-1] / sma50 - 1.0))

        # 15: Kronos entropy (if dist provided)
        if dist is not None:
            entropy = float(dist.entropy())
        else:
            entropy = 1.5  # Default mid-range entropy
        features.append(entropy)

        return features

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        """Compute RSI (Relative Strength Index).

        Args:
            close: Close prices (at least period+1 values)
            period: RSI period (default 14)

        Returns:
            RSI value (0-100)
        """
        if len(close) < period + 1:
            return 50.0

        deltas = np.diff(close[-period - 1:])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)

    @staticmethod
    def _compute_atr_ratio_gbs(high: np.ndarray, low: np.ndarray,
                               close: np.ndarray, period: int = 14) -> float:
        """Compute ATR ratio: ATR(14) / current price."""
        if len(close) < period + 1:
            return 0.01

        tr_values = []
        for i in range(1, len(close)):
            h_l = high[i] - low[i]
            h_c = abs(high[i] - close[i - 1])
            l_c = abs(low[i] - close[i - 1])
            tr = max(h_l, h_c, l_c)
            tr_values.append(tr)

        if len(tr_values) < period:
            return 0.01

        atr = np.mean(tr_values[-period:])
        if close[-1] > 0:
            return float(atr / close[-1])
        return 0.01

    @staticmethod
    def _compute_volume_zscore(volume: np.ndarray, period: int = 20) -> float:
        """Compute volume z-score (current vs. 20-day SMA)."""
        if len(volume) < period + 1:
            return 0.0

        recent_vol = volume[-period:]
        mean_vol = np.mean(recent_vol)
        std_vol = np.std(recent_vol)

        if std_vol < 1e-6:
            return 0.0

        return float((volume[-1] - mean_vol) / std_vol)

    @staticmethod
    def _compute_day_of_week_onehot(history: pd.DataFrame) -> list:
        """Compute day-of-week one-hot encoding (5 features for Mon-Fri).

        Args:
            history: DataFrame with datetime index

        Returns:
            List of 5 binary features (one-hot for Mon-Fri)
        """
        if not isinstance(history.index, pd.DatetimeIndex):
            return [0.0] * 5

        dow = history.index[-1].dayofweek  # 0=Mon, 4=Fri
        one_hot = [0.0] * 5

        if 0 <= dow <= 4:
            one_hot[dow] = 1.0

        return one_hot

    @staticmethod
    def _compute_rolling_volatility(close: np.ndarray,
                                     period: int = 20) -> float:
        """Compute rolling volatility (std of log returns)."""
        if len(close) < period + 1:
            return 0.01

        log_ret = np.diff(np.log(close[-period:]))
        volatility = float(np.std(log_ret))
        return max(volatility, 0.001)  # Ensure minimum value


# =============================================================================
# LPPLS Bubble Detection (Sornette)
# =============================================================================

def fit_lppls(log_prices: np.ndarray, n_starts: int = 8, seed: int = 0) -> Dict[str, Any]:
    """Fit log-periodic power law singularity (LPPLS) model via multi-start Nelder-Mead.

    Model: ln p(t) = A + B(tc-t)^m + C1(tc-t)^m cos(ω ln(tc-t)) + C2(tc-t)^m sin(ω ln(tc-t))

    Nonlinear 3-parameter reduction: optimize (tc, m, ω) via Nelder-Mead, profile out
    (A, B, C1, C2) via linear least squares at each step.

    Args:
        log_prices: Log prices (at least 50 points), indexed t=0,1,...,T-1
        n_starts: Number of random starting points for Nelder-Mead
        seed: Random seed for reproducible starts

    Returns:
        Dictionary with keys:
        - "tc": critical time (optimal tc)
        - "m": power law exponent
        - "omega": angular frequency
        - "resid_rms": RMS residual of fit
        - "converged": bool, True if optimization succeeded
        - "bubble": bool, True iff bubble signature (m ∈ [0.1, 0.9], ω ∈ [6, 13],
                    tc ∈ (T, T+60], B < 0, converged)
        - "B": slope coefficient (negative for super-exponential up)
    """
    T = len(log_prices)
    if T < 50:
        return {
            "tc": float(T), "m": 0.5, "omega": 8.0,
            "resid_rms": np.inf, "converged": False, "bubble": False, "B": 0.0
        }

    t_array = np.arange(T, dtype=float)
    y = log_prices

    rng = np.random.default_rng(seed)

    # Multi-start Nelder-Mead
    best_loss = np.inf
    best_params = None

    for start_idx in range(n_starts):
        # Random starting point: tc in (T, T+60), m in [0.1, 0.9], ω in [6, 13]
        tc_start = float(T + rng.uniform(1, 60))
        m_start = float(rng.uniform(0.1, 0.9))
        omega_start = float(rng.uniform(6, 13))

        x0 = np.array([tc_start, m_start, omega_start])

        try:
            result = minimize(
                _lppls_objective,
                x0,
                args=(t_array, y),
                method="Nelder-Mead",
                options={"maxiter": 200, "xatol": 1e-6, "fatol": 1e-9}
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x
        except (ValueError, RuntimeError):
            continue

    if best_params is None:
        return {
            "tc": float(T), "m": 0.5, "omega": 8.0,
            "resid_rms": np.inf, "converged": False, "bubble": False, "B": 0.0
        }

    # Unpack best params and fit linear coefficients
    tc, m, omega = best_params
    tc = float(tc)
    m = float(m)
    omega = float(omega)

    # Fit linear coefficients A, B, C1, C2 via least squares
    X = _build_lppls_design_matrix(t_array, tc, m, omega)
    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        A, B, C1, C2 = coeffs
    except (np.linalg.LinAlgError, ValueError):
        A, B, C1, C2 = 0.0, 0.0, 0.0, 0.0

    # Compute residuals
    y_pred = X @ coeffs
    residuals = y - y_pred
    resid_rms = float(np.sqrt(np.mean(residuals ** 2)))

    # Check bubble signature
    converged = best_loss < np.inf
    bubble = (
        converged
        and 0.1 <= m <= 0.9
        and 6.0 <= omega <= 13.0
        and T < tc <= T + 60.0
        and B < 0.0
    )

    return {
        "tc": tc,
        "m": m,
        "omega": omega,
        "resid_rms": resid_rms,
        "converged": converged,
        "bubble": bubble,
        "B": float(B),
    }


def _lppls_objective(params: np.ndarray, t_array: np.ndarray,
                     y: np.ndarray) -> float:
    """Objective function for LPPLS: squared error with profiled linear coefficients.

    Args:
        params: [tc, m, omega]
        t_array: Time indices
        y: Log prices

    Returns:
        Squared error after profiling out A, B, C1, C2
    """
    tc, m, omega = params

    # Guard against invalid params
    if tc <= np.max(t_array) or m <= 0 or omega <= 0:
        return 1e10

    try:
        X = _build_lppls_design_matrix(t_array, tc, m, omega)
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ coeffs
        sse = float(np.sum((y - y_pred) ** 2))
        return sse
    except (ValueError, np.linalg.LinAlgError):
        return 1e10


def _build_lppls_design_matrix(t_array: np.ndarray, tc: float, m: float,
                               omega: float) -> np.ndarray:
    """Build design matrix X for LPPLS model with given (tc, m, omega).

    Columns: [1, (tc-t)^m, (tc-t)^m cos(ω ln(tc-t)), (tc-t)^m sin(ω ln(tc-t))]

    Args:
        t_array: Time array
        tc: Critical time
        m: Power law exponent
        omega: Angular frequency

    Returns:
        Design matrix of shape (len(t_array), 4)
    """
    tau = tc - t_array  # tc - t for each t

    # Guard against negative tau
    if np.any(tau <= 0):
        raise ValueError("tc must be > all time indices")

    tau_m = tau ** m
    cos_term = tau_m * np.cos(omega * np.log(tau))
    sin_term = tau_m * np.sin(omega * np.log(tau))

    X = np.column_stack([
        np.ones_like(t_array),  # A
        tau_m,                  # B * (tc-t)^m
        cos_term,               # C1 * (tc-t)^m cos(ω ln(tc-t))
        sin_term                # C2 * (tc-t)^m sin(ω ln(tc-t))
    ])

    return X


# =============================================================================
# LPPLS Guard Strategy
# =============================================================================

class LPPLSGuardStrategy(Strategy):
    """LPPLS bubble detection wrapper around any base strategy.

    Fits the log-periodic power law singularity model on trailing log-prices.
    When bubble signature detected:
    - Vetoes new LONG entries (returns None)
    - Passes SHORT signals through, boosting confidence by 1.2x (capped at 1.0)

    When no bubble: pass-through unchanged.
    When history < lookback: pass-through unchanged.

    Parameters:
        base_strategy: Strategy to wrap
        lookback: Window for LPPLS fit (default 250)
        refit_days: Refit every N calls (default 10)
        n_starts: Number of Nelder-Mead starts (default 8, reduce to 4 in tests)
        seed: Random seed for reproducibility
    """

    name = "lppls_guard"

    def __init__(self, base_strategy: Strategy, lookback: int = 250,
                 refit_days: int = 10, n_starts: int = 8, seed: int = 0):
        self.base_strategy = base_strategy
        self.lookback = lookback
        self.refit_days = refit_days
        self.n_starts = n_starts
        self.seed = seed

        # State tracking
        self.call_count = 0
        self.fit_count = 0
        self.last_fit = None  # Dict from fit_lppls

    def reset(self) -> None:
        """Reset state for walk-forward folds."""
        self.call_count = 0
        self.fit_count = 0
        self.last_fit = None

    def generate_signal(self, dist, current_price: float, history: pd.DataFrame,
                        context: Dict[str, Any], **kwargs) -> Optional[Signal]:
        """Generate signal via base strategy, filtered by LPPLS bubble detection.

        Args:
            dist: KairosDistribution
            current_price: Current price
            history: Historical OHLCV data
            context: Context dict

        Returns:
            Signal (potentially modified) or None
        """
        self.call_count += 1

        # Get base signal (may be None)
        base_sig = self.base_strategy.generate_signal(dist, current_price,
                                                      history, context, **kwargs)

        # Refit every refit_days calls
        if self.call_count % self.refit_days == 0:
            self._refit_lppls(history)

        # Insufficient history -> pass-through
        if len(history) < self.lookback:
            return base_sig

        # No fit -> pass-through
        if self.last_fit is None:
            return base_sig

        bubble_detected = self.last_fit.get("bubble", False)

        # No bubble -> pass-through
        if not bubble_detected:
            return base_sig

        # Bubble detected -> apply guard logic
        if base_sig is None:
            return None

        # Veto LONG entries during bubble
        if base_sig.direction == Direction.LONG:
            return None

        # SHORT signals: boost confidence by 1.2x, capped at 1.0
        if base_sig.direction == Direction.SHORT:
            boosted_confidence = min(base_sig.confidence * 1.2, 1.0)
            return Signal(
                direction=base_sig.direction,
                size=base_sig.size,
                entry=base_sig.entry,
                stop=base_sig.stop,
                target=base_sig.target,
                strategy_name=base_sig.strategy_name,
                confidence=boosted_confidence,
                expected_value=base_sig.expected_value,
                metadata={**base_sig.metadata, "lppls_bubble": True,
                          "lppls_fit": self.last_fit}
            )

        # Fallback (shouldn't reach)
        return base_sig

    def _refit_lppls(self, history: pd.DataFrame) -> None:
        """Refit LPPLS model on trailing lookback log-prices.

        Args:
            history: Historical OHLCV data
        """
        if len(history) < self.lookback:
            self.last_fit = None
            return

        # Extract trailing log prices
        closes = history["close"].values[-self.lookback:]
        log_prices = np.log(closes)

        # Fit LPPLS
        self.last_fit = fit_lppls(log_prices, n_starts=self.n_starts, seed=self.seed)
        self.fit_count += 1


# =============================================================================
# ML Bracket Strategy (E18-S04)
# =============================================================================

class MLBracketStrategy(Strategy):
    """Per-candidate stop/target selector using trained GBM classifiers.

    Wraps a base strategy and, when the base emits a signal, evaluates the
    expected value (EV) of each trained TP/SL candidate ({stop_pct, target_pct}
    pair) via its corresponding GradientBoostedStumps classifier. Selects the
    argmax-EV candidate and returns a new Signal with overridden stop/target.

    Requires trained classifiers saved by E18-S03's `train_tpsl_model.py` at
    `model_dir`. Raises a clear error at construction if no models are found.

    Args:
        base_strategy: Strategy instance to wrap.
        model_dir: Path to directory containing trained models (default
                   `data/tpsl_models/`).
        interval: Bar interval string (e.g. "1d", "1h") this strategy instance
                  is evaluating. Passed at construction rather than read from
                  context -- no real context builder (kairos_orchestrator.py's
                  `_run_day`, kairos_signals.py's `_build_context`) ever sets
                  an "interval" key, and the interval is fixed per
                  orchestrator run anyway (unlike ticker, which varies call to
                  call and IS available from context -- see `current_symbol`
                  below).

    Raises:
        FileNotFoundError: If model_dir does not exist or contains no trained
                          classifiers.
    """

    name = "ml_bracket"

    def __init__(self, base_strategy: Strategy, model_dir: str = "data/tpsl_models/",
                 interval: str = "1d"):
        self.base_strategy = base_strategy
        self.model_dir = model_dir
        self.interval = interval

        # Load feature metadata
        metadata_path = os.path.join(model_dir, "feature_metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No trained classifiers found at {model_dir}: "
                f"missing {metadata_path}. Run train_tpsl_model.py first."
            )

        with open(metadata_path, "r") as fh:
            metadata = json.load(fh)

        self.feature_columns = metadata["feature_columns"]
        self.asset_classes = metadata["asset_classes"]
        self.interval_vocab = metadata["interval_vocab"]
        self.candidates_meta = metadata["candidates"]

        # Load trained classifiers
        self.models: Dict[str, GradientBoostedStumps] = {}
        for key in self.candidates_meta.keys():
            model_path = os.path.join(model_dir, f"{key}.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Trained classifier not found at {model_path}. "
                    f"Run train_tpsl_model.py first."
                )
            with open(model_path, "rb") as fh:
                self.models[key] = pickle.load(fh)

        if not self.models:
            raise FileNotFoundError(
                f"No trained classifiers found at {model_dir}. "
                f"Run train_tpsl_model.py first."
            )

    def generate_signal(self, dist, current_price: float, history: pd.DataFrame,
                        context: Dict[str, Any], **kwargs) -> Optional[Signal]:
        """Generate signal via base strategy, override stop/target via ML classifiers.

        Args:
            dist: KairosDistribution (unused, available for future features)
            current_price: Current price (entry point)
            history: Historical OHLCV data
            context: Context dict. Must contain "current_symbol" (the ticker
                     key every real context builder actually sets -- neither
                     kairos_orchestrator.py's `_run_day` nor kairos_signals.py's
                     `_build_context` ever sets a "ticker" key). "date", if
                     present, is used as the feature-extraction as-of date
                     (see below); interval comes from `self.interval`, not
                     context.

        Returns:
            Signal with ML-selected stop/target, or None if base returns None or
            feature extraction fails.
        """
        # Import here to avoid circular dependency (kairos_ml -> kairos_tpsl_features
        # -> kairos_strategies -> kairos_orchestrator -> kairos_ml)
        from kairos_tpsl_features import extract_features  # noqa: E402

        # Call base strategy
        base_sig = self.base_strategy.generate_signal(dist, current_price,
                                                      history, context, **kwargs)
        if base_sig is None:
            return None

        # Only LONG/SHORT signals have a directional bracket to optimize.
        # Several base strategies (kairos_crypto.py, kairos_forex.py,
        # kairos_meta.py, kairos_path.py) return an explicit Direction.FLAT
        # Signal rather than None. Scoring FLAT with the SHORT-branch stop/
        # target formula below would stamp a nonsensical bracket into the
        # metadata even though direction stays FLAT -- skip ML selection
        # entirely and pass through unchanged, same as the missing-context
        # fallback below. This check must run before the context lookup so a
        # FLAT signal always passes through regardless of context contents.
        if base_sig.direction not in (Direction.LONG, Direction.SHORT):
            return base_sig

        # Ticker from context: "current_symbol" is the key every real context
        # builder sets; "ticker" (the old key) is never set by either, so
        # reading that would silently never fire once this strategy is
        # registered live (context.get would always be None).
        ticker = context.get("current_symbol")
        if ticker is None:
            # Missing required context, pass through unchanged
            return base_sig

        # "As of" date for feature extraction: the real current date, not
        # history.index[-1]. Under prediction_usage_mode="distribution_as_bar"
        # (kairos_prediction_usage.py), history's last row can be a synthetic
        # forecast bar the orchestrator appended one interval past today --
        # using it as as_of would defeat extract_features()'s own no-lookahead
        # truncation (`history[history.index <= as_of]`) and let ATR/
        # realized_vol/trend/range-position read the model's own forecast as
        # if it were real history (this project has been bitten by this exact
        # lookahead-via-synthetic-bar bug class twice before -- see
        # CLAUDE.md's naive-baseline/oracle-peek sections). context["date"] is
        # set by every real context builder to the actual date being
        # processed, which is always <= the last real bar and strictly before
        # any appended synthetic one. Falls back to history.index[-1] only
        # when a caller doesn't supply "date" (e.g. a test), matching this
        # strategy's pre-fix behavior in that case.
        as_of_source = context.get("date", history.index[-1])
        as_of_date = pd.Timestamp(as_of_source).date().isoformat()

        # Extract features (price-history only, matching training)
        try:
            features = extract_features(
                ticker=ticker,
                as_of=as_of_date,
                interval=self.interval,
                entry=base_sig.entry,
                history=history,
            )
        except ValueError:
            # Insufficient history or other extraction failure, pass through
            return base_sig

        # Vectorize features in exact order from feature_metadata.json
        X = self._vectorize_features(features)

        # Evaluate only the candidates trained for (or applicable to) this
        # signal's own asset class -- see _select_candidate_keys() for why an
        # unfiltered loop over self.models lets an out-of-distribution
        # classifier (e.g. a "..._crypto" model scoring an equity signal)
        # win the argmax.
        candidate_keys = self._select_candidate_keys(features["asset_class"])

        # Evaluate all candidates
        best_ev = float("-inf")
        best_key = None
        best_stop = None
        best_target = None
        best_p_win = 0.0

        for key in candidate_keys:
            model = self.models[key]
            # Get stop_pct, target_pct from metadata
            meta = self.candidates_meta[key]
            stop_pct = meta["stop_pct"]
            target_pct = meta["target_pct"]

            # Compute absolute stop/target from base signal's entry
            if base_sig.direction == Direction.LONG:
                candidate_stop = base_sig.entry * (1.0 - stop_pct / 100.0)
                candidate_target = base_sig.entry * (1.0 + target_pct / 100.0)
            else:  # SHORT (FLAT/other directions already returned above)
                candidate_stop = base_sig.entry * (1.0 + stop_pct / 100.0)
                candidate_target = base_sig.entry * (1.0 - target_pct / 100.0)

            # Predict P(win) for this candidate. Only catch genuinely-expected
            # failures here: the model not being fitted, or a feature-vector
            # shape mismatch between this candidate's saved .pkl and the
            # shared feature_metadata.json (e.g. a partially regenerated
            # data/tpsl_models/) -- both surface as ValueError/IndexError from
            # GradientBoostedStumps.predict_proba. Anything else is a real bug
            # and must propagate rather than silently degrade to "no ML edge
            # found", indistinguishable from a legitimate low-EV result.
            try:
                p_win = float(model.predict_proba(X.reshape(1, -1))[0])
            except (ValueError, IndexError):
                continue

            # Compute EV: p_win * reward_pct - (1 - p_win) * risk_pct
            risk_pct = abs(candidate_stop - base_sig.entry) / base_sig.entry * 100
            reward_pct = abs(candidate_target - base_sig.entry) / base_sig.entry * 100
            ev = p_win * reward_pct - (1.0 - p_win) * risk_pct

            # Track best EV
            if ev > best_ev:
                best_ev = ev
                best_key = key
                best_stop = candidate_stop
                best_target = candidate_target
                best_p_win = p_win

        # If no valid candidate was found, pass through unchanged
        if best_key is None or best_stop is None or best_target is None:
            return base_sig

        # Get original stop/target for auditability
        original_stop = base_sig.stop
        original_target = base_sig.target

        # Get original risk/reward percentages
        orig_risk_pct = abs(original_stop - base_sig.entry) / base_sig.entry * 100
        orig_reward_pct = abs(original_target - base_sig.entry) / base_sig.entry * 100

        # Build new Signal with ML-selected brackets
        best_risk_pct = abs(best_stop - base_sig.entry) / base_sig.entry * 100
        best_reward_pct = abs(best_target - base_sig.entry) / base_sig.entry * 100

        # confidence/expected_value must be recomputed from the CHOSEN
        # candidate, not copied from base_sig -- base_sig's values describe
        # the ORIGINAL (now-discarded) stop/target. kairos_signals.py's
        # EV-based sort/gate (_ev_pct_value, min_ev_pct filtering) reads
        # sig.expected_value directly, so a stale value would rank/gate a
        # trade that no longer exists. expected_value is stored in absolute
        # price units codebase-wide (kairos_signals._ev_pct_value recovers
        # the percent via `expected_value / entry * 100`); best_ev above is
        # already a percent (computed from risk_pct/reward_pct), so convert
        # it back to price units for the Signal field.
        recomputed_expected_value = (best_ev / 100.0) * base_sig.entry

        return Signal(
            direction=base_sig.direction,
            size=base_sig.size,
            entry=base_sig.entry,
            stop=best_stop,
            target=best_target,
            strategy_name=self.name,
            confidence=best_p_win,
            expected_value=recomputed_expected_value,
            metadata={
                **base_sig.metadata,
                "ml_bracket": {
                    "original_stop": original_stop,
                    "original_target": original_target,
                    "original_stop_pct": orig_risk_pct,
                    "original_target_pct": orig_reward_pct,
                    "chosen_stop": best_stop,
                    "chosen_target": best_target,
                    "chosen_stop_pct": best_risk_pct,
                    "chosen_target_pct": best_reward_pct,
                    "predicted_p_win": best_p_win,
                    "predicted_ev": best_ev,
                    "candidate_key": best_key,
                }
            }
        )

    def _select_candidate_keys(self, asset_class: str) -> list[str]:
        """Resolve one usable model key per (stop_pct, target_pct) grid combo.

        `train_tpsl_model.py` trains either one pooled model per combo (key
        `"{stop_pct}_{target_pct}"`, `meta["mode"] == "pooled"`) or, when there
        was enough per-class training data (`check_class_sufficiency()`), one
        model per (combo, asset class) instead (key
        `"{stop_pct}_{target_pct}_{cls}"`, `meta["mode"] == "per_class"`,
        `meta["asset_class"] == cls`) -- never both for the same combo.
        Scoring a signal with a per-class model trained on a DIFFERENT class
        (e.g. an equity signal against a "..._crypto" model) is an
        out-of-distribution prediction that can still win the EV argmax, so
        for each combo this picks, in order: the per-class model matching
        `asset_class`, else the pooled model for that combo, else nothing
        (that combo has no usable model for this signal and is skipped).

        Args:
            asset_class: This signal's own asset class (from extract_features()).

        Returns:
            List of `self.models` keys usable for this asset class, at most
            one per (stop_pct, target_pct) combo.
        """
        by_combo: dict[tuple[float, float], dict[str, str]] = {}
        for key, meta in self.candidates_meta.items():
            combo = (meta["stop_pct"], meta["target_pct"])
            slot = by_combo.setdefault(combo, {})
            if meta.get("mode") == "per_class":
                if meta.get("asset_class") == asset_class:
                    slot["match"] = key
            else:  # pooled
                slot["pooled"] = key

        selected: list[str] = []
        for options in by_combo.values():
            if "match" in options:
                selected.append(options["match"])
            elif "pooled" in options:
                selected.append(options["pooled"])
        return selected

    def _vectorize_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert extracted feature dict to numeric vector in feature column order.

        Args:
            features: Dict from extract_features() with keys:
                     atr, realized_vol, asset_class, interval, trend_10, range_position

        Returns:
            1D numpy array of length len(self.feature_columns), one-hot encoded
            for categorical features.

        Raises:
            ValueError: if features["asset_class"] or features["interval"] is
                not present in the saved training vocabulary
                (self.asset_classes / self.interval_vocab). train_tpsl_model.py's
                own `_feature_columns()` docstring calls a silent mismatch here
                "a silent-failure risk (wrong columns score without raising)":
                leaving the one-hot segment all-zero produces a real
                out-of-distribution feature vector that would be fed straight
                into predict_proba() looking identical to a legitimate,
                confidently-low-EV prediction.
        """
        if features["asset_class"] not in self.asset_classes:
            raise ValueError(
                f"Unseen asset_class {features['asset_class']!r} not in training "
                f"vocabulary {self.asset_classes}; refusing to score with an "
                f"all-zero one-hot segment"
            )
        if features["interval"] not in self.interval_vocab:
            raise ValueError(
                f"Unseen interval {features['interval']!r} not in training "
                f"vocabulary {self.interval_vocab}; refusing to score with an "
                f"all-zero one-hot segment"
            )

        X = np.zeros(len(self.feature_columns), dtype=float)

        # Map feature name -> column index
        for col_idx, col_name in enumerate(self.feature_columns):
            if "=" in col_name:
                # Categorical one-hot column (e.g., "asset_class=equity")
                category, value = col_name.split("=", 1)
                if category == "asset_class":
                    if features["asset_class"] == value:
                        X[col_idx] = 1.0
                elif category == "interval":
                    if features["interval"] == value:
                        X[col_idx] = 1.0
            else:
                # Numeric feature
                if col_name in features:
                    X[col_idx] = features[col_name]

        return X
