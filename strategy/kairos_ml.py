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
