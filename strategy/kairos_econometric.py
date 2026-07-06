"""
Econometric strategies: ARIMA disagreement filter, VAR lead-lag, seasonality, etc.

No statsmodels dependency. All fits via numpy least squares + scipy optimizers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from kairos_backtest import Strategy, Signal, Direction, KairosDistribution


# =============================================================================
# HELPERS
# =============================================================================

def _lagged_ols(y: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
    """
    Fit y ~ X via ordinary least squares, return regression diagnostics.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Dependent variable.
    X : np.ndarray, shape (n, k)
        Design matrix (should include intercept column if needed).

    Returns
    -------
    dict with keys:
        "coef": np.ndarray, shape (k,), OLS coefficients
        "se": np.ndarray, shape (k,), standard errors
        "tstats": np.ndarray, shape (k,), t-statistics
        "resid": np.ndarray, shape (n,), residuals
        "aic": float, Akaike Information Criterion
    """
    # Solve via normal equations: X'X coef = X'y
    # lstsq is more numerically stable
    n, k = X.shape
    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # Residuals from the fit
    resid = y - X @ coef

    # Residual standard error (unbiased estimate)
    if n > k:
        sigma2 = np.sum(resid**2) / (n - k)
    else:
        sigma2 = np.sum(resid**2) / max(n, 1)

    # Covariance matrix of coefficients
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        cov_coef = sigma2 * XtX_inv
    except np.linalg.LinAlgError:
        # Singular design matrix; return nans for SEs
        cov_coef = np.full((k, k), np.nan)

    se = np.sqrt(np.diag(cov_coef))
    tstats = coef / (se + 1e-10)  # Avoid division by zero

    # AIC = n*ln(RSS/n) + 2k
    rss = np.sum(resid**2)
    aic = n * np.log(rss / n + 1e-10) + 2 * k

    return {
        "coef": coef,
        "se": se,
        "tstats": tstats,
        "resid": resid,
        "aic": aic,
    }


# =============================================================================
# ARIMA DISAGREEMENT FILTER
# =============================================================================

class ARIMADisagreementStrategy(Strategy):
    """
    Filter wrapper that vetoes signals when AR(p) and Kronos disagree on direction.

    Fits AR(p) with drift on trailing closes; selects p by AIC over range 1..max_p.
    If the ARIMA point forecast direction disagrees with Kronos mean forecast
    direction, returns None (veto). If they agree, boosts confidence by agree_boost
    (capped at 1.0).

    Parameters
    ----------
    base_strategy : Strategy
        The wrapped strategy to filter.
    lookback : int, default 120
        Number of trailing closes to use for AR fit.
    max_p : int, default 5
        Maximum AR order to evaluate in AIC selection (test 1..max_p).
    agree_boost : float, default 1.2
        Multiplier for confidence when AR and Kronos agree (capped at 1.0).
    """
    name = "arima_disagreement"

    def __init__(self, base_strategy: Strategy, lookback: int = 120,
                 max_p: int = 5, agree_boost: float = 1.2):
        self.base_strategy = base_strategy
        self.lookback = lookback
        self.max_p = max_p
        self.agree_boost = agree_boost

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        """
        Wrap the base strategy and apply ARIMA disagreement check.

        Returns None (veto) if AR forecast and Kronos mean disagree in sign.
        Otherwise multiplies confidence by agree_boost if they agree.
        """
        # Get the base signal first
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context, **kwargs)
        if base_signal is None:
            return None

        # Extract trailing closes for AR fit
        if history is None or len(history) < self.lookback + 1:
            # Not enough history; pass through
            return base_signal

        closes = history["close"].tail(self.lookback + 1).values
        if len(closes) < self.lookback + 1:
            return base_signal

        # Fit AR(p) and select best p by AIC
        best_p = self._select_ar_order(closes)
        if best_p is None:
            return base_signal

        # Compute AR point forecast for next day
        ar_forecast = self._ar_forecast(closes, best_p)
        if ar_forecast is None:
            return base_signal

        # Compare directions: AR forecast vs. Kronos mean
        kronos_mean = dist.stats["close"]["mean"]

        ar_direction = np.sign(ar_forecast - current_price)
        kronos_direction = np.sign(kronos_mean - current_price)

        # If directions disagree, veto
        if ar_direction != kronos_direction and ar_direction != 0 and kronos_direction != 0:
            return None

        # If they agree, boost confidence
        if ar_direction == kronos_direction and ar_direction != 0:
            boosted_conf = min(base_signal.confidence * self.agree_boost, 1.0)
            return Signal(
                direction=base_signal.direction,
                size=base_signal.size,
                entry=base_signal.entry,
                stop=base_signal.stop,
                target=base_signal.target,
                strategy_name=f"{base_signal.strategy_name}+arima_agree",
                confidence=boosted_conf,
                expected_value=base_signal.expected_value,
                metadata=base_signal.metadata,
            )

        # Neutral case or no boost; pass through
        return base_signal

    def _select_ar_order(self, closes: np.ndarray) -> Optional[int]:
        """
        Select AR order 1..max_p by minimizing AIC.

        Returns
        -------
        int or None
            The selected order, or None if fit fails.
        """
        aic_scores = []
        for p in range(1, self.max_p + 1):
            fit_result = self._fit_ar(closes, p)
            if fit_result is None:
                continue
            aic_scores.append((p, fit_result["aic"]))

        if not aic_scores:
            return None

        best_p = min(aic_scores, key=lambda x: x[1])[0]
        return best_p

    def _fit_ar(self, closes: np.ndarray, p: int) -> Optional[Dict[str, Any]]:
        """
        Fit AR(p) with drift: closes_t = c + phi_1*closes_{t-1} + ... + phi_p*closes_{t-p} + e_t

        Returns
        -------
        dict or None
            The result from _lagged_ols, or None if fit fails.
        """
        n = len(closes)
        if n <= p + 1:
            return None

        # Build design matrix: intercept + lags 1..p
        X = np.ones((n - p, p + 1))
        y = closes[p:]

        for lag in range(1, p + 1):
            X[:, lag] = closes[p - lag : -lag or None]

        try:
            result = _lagged_ols(y, X)
            return result
        except Exception:
            return None

    def _ar_forecast(self, closes: np.ndarray, p: int) -> Optional[float]:
        """
        Forecast next close using fitted AR(p).

        Returns
        -------
        float or None
            The point forecast, or None if fit fails.
        """
        fit_result = self._fit_ar(closes, p)
        if fit_result is None:
            return None

        coef = fit_result["coef"]
        # forecast = c + phi_1*closes[-1] + ... + phi_p*closes[-(p)]
        forecast = coef[0]  # intercept
        for lag in range(1, p + 1):
            forecast += coef[lag] * closes[-lag]

        return forecast


# =============================================================================
# VAR LEAD-LAG STRATEGY
# =============================================================================

class VARLeadLagStrategy(Strategy):
    """
    Standalone VAR(1) lead-lag strategy detecting cross-asset influences.

    Fits a VAR(1) model on the returns panel to detect when asset j's lagged
    return significantly (|t|>2) predicts asset i's return. If yesterday's
    j-move implies an i-move that agrees with Kronos direction, emits a signal.

    Context requirements:
        - context["returns_window"]: DataFrame of daily returns (columns=symbols, rows=dates)
        - context["symbol"]: The target symbol being evaluated

    Parameters
    ----------
    stop_pct : float, default 15.0
        Stop-loss percentile (for longs; reversed for shorts).
    target_pct : float, default 85.0
        Take-profit percentile (for longs; reversed for shorts).
    t_stat_threshold : float, default 2.0
        Threshold for significance of t-statistic on lagged coefficients.
    lookback : int, default 60
        Minimum number of rows required in returns_window.
    """
    name = "var_leadlag"

    def __init__(self, stop_pct: float = 15.0, target_pct: float = 85.0,
                 t_stat_threshold: float = 2.0, lookback: int = 60):
        self.stop_pct = int(stop_pct)
        self.target_pct = int(target_pct)
        self.t_stat_threshold = t_stat_threshold
        self.lookback = lookback

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        """
        Fit VAR(1) on returns panel and check for lead-lag relationships.

        Returns Signal if a significant lead-lag from another asset agrees
        with Kronos direction, else None.
        """
        # Validate context
        if context is None or "returns_window" not in context or "symbol" not in context:
            return None

        returns_window = context.get("returns_window")
        symbol = context.get("symbol")

        # Validate returns_window
        if returns_window is None or len(returns_window) < self.lookback:
            return None

        # Check that symbol is in the returns window
        if symbol not in returns_window.columns:
            return None

        try:
            # Extract the target return series
            y = returns_window[symbol].values  # Shape (n,)

            if len(y) < 2:
                return None

            n_obs, n_assets = returns_window.shape

            # Build regression for VAR(1): r_i,t = c + Σ_j β_j*r_j,t-1 + e_t
            # y_reg: returns for t=1..n-1
            y_reg = y[1:]

            # Design matrix: [intercept, lagged_returns_t-1]
            X = np.ones((n_obs - 1, 1 + n_assets))
            X[:, 1:] = returns_window.iloc[:-1, :].values  # Lagged returns at t-1

            # Fit via OLS
            fit_result = _lagged_ols(y_reg, X)

            coef = fit_result["coef"]  # Shape (1 + n_assets,)
            tstats = fit_result["tstats"]  # Shape (1 + n_assets,)

        except Exception:
            return None

        # Check for significant lead-lag relationships
        kronos_mean = dist.stats["close"]["mean"]
        kronos_direction = np.sign(kronos_mean - current_price)

        # If Kronos is neutral, no trade
        if kronos_direction == 0:
            return None

        # Look for significant j (other assets) that imply agreement
        # Index 0 is intercept, indices 1..n_assets are the lagged returns
        agreement_found = False
        max_abs_t = 0.0

        for j in range(n_assets):
            t_stat = tstats[1 + j]  # t-stat for the j-th lagged return coefficient

            if abs(t_stat) > self.t_stat_threshold:
                # Compute implied move: yesterday's j-return × coefficient
                yesterday_return_j = returns_window.iloc[-2, j]
                coef_j = coef[1 + j]
                implied_move = coef_j * yesterday_return_j
                implied_direction = np.sign(implied_move)

                # Check if implied direction agrees with Kronos
                if implied_direction == kronos_direction and implied_direction != 0:
                    agreement_found = True
                    max_abs_t = max(max_abs_t, abs(t_stat))

        if not agreement_found:
            return None

        # Emit signal with agreement
        if kronos_direction > 0:
            # LONG
            direction = Direction.LONG
            stop = dist.stats["close"][f"pct_{self.stop_pct}"]
            target = dist.stats["close"][f"pct_{self.target_pct}"]
        else:
            # SHORT
            direction = Direction.SHORT
            stop = dist.stats["close"][f"pct_{self.target_pct}"]
            target = dist.stats["close"][f"pct_{self.stop_pct}"]

        # Calculate expected value and position size
        ev = dist.expected_value(current_price, target, stop)
        if ev <= 0:
            return None

        kelly = dist.kelly_fraction(current_price, target, stop)
        size = min(kelly * 0.5, 1.0)

        # Confidence from t-stat (scale to [0, 1])
        # Use sigmoid: confidence approaches 1 as |t| increases
        confidence = max_abs_t / (max_abs_t + self.t_stat_threshold)
        confidence = min(max(confidence, 0.0), 1.0)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={"max_abs_t": float(max_abs_t)},
        )
