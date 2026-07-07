"""
Econometric strategies: ARIMA disagreement filter, VAR lead-lag, seasonality,
changepoint detection, etc.

No statsmodels dependency. All fits via numpy least squares + scipy optimizers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, Tuple
from scipy.special import loggamma
from scipy import stats
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


def granger_f_test(y: np.ndarray, x: np.ndarray, max_lag: int = 3) -> Dict[str, Any]:
    """
    Perform rolling Granger F-test between y and x.

    Fits restricted model (y on its own lags 1..L) and unrestricted model
    (adding x's lags 1..L) for each L in 1..max_lag. Computes F-statistic
    and p-value for each lag, returns results for the lag with minimum p-value.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Dependent variable (follower).
    x : np.ndarray, shape (n,)
        Predictor variable (leader).
    max_lag : int, default 3
        Maximum lag to evaluate (test lags 1..max_lag).

    Returns
    -------
    dict with keys:
        "f_stat": float, F-statistic at best_lag
        "p_value": float, p-value via scipy.stats.f.sf
        "best_lag": int, lag with minimum p-value (1..max_lag)
        "coef_sign": float, sign of sum of x-lag coefficients at best_lag
    """
    n = len(y)
    best_result = {
        "f_stat": 0.0,
        "p_value": 1.0,
        "best_lag": 1,
        "coef_sign": 0.0,
    }

    for lag in range(1, max_lag + 1):
        # Require at least lag + 1 observations
        if n <= lag:
            continue

        # Build restricted model: y_t = c + phi_1*y_{t-1} + ... + phi_L*y_{t-L} + e_t
        n_reg = n - lag  # Number of observations for regression
        X_r = np.ones((n_reg, lag + 1))  # Intercept + lag y's
        y_reg = y[lag:]

        for i in range(1, lag + 1):
            X_r[:, i] = y[lag - i : -i or None]

        # Fit restricted model
        try:
            fit_r = _lagged_ols(y_reg, X_r)
            rss_r = np.sum(fit_r["resid"] ** 2)
        except Exception:
            continue

        # Build unrestricted model: y_t = c + phi_1*y_{t-1} + ... + phi_L*y_{t-L} +
        #                                  beta_1*x_{t-1} + ... + beta_L*x_{t-L} + e_t
        X_u = np.ones((n_reg, lag + 1 + lag))  # Intercept + lag y's + lag x's
        X_u[:, 1 : lag + 1] = X_r[:, 1:]  # Copy lagged y's

        for i in range(1, lag + 1):
            X_u[:, lag + i] = x[lag - i : -i or None]

        # Fit unrestricted model
        try:
            fit_u = _lagged_ols(y_reg, X_u)
            rss_u = np.sum(fit_u["resid"] ** 2)
        except Exception:
            continue

        # Compute F-statistic
        # F = ((RSS_r - RSS_u) / L) / (RSS_u / (n_reg - 2*lag - 1))
        numerator = (rss_r - rss_u) / lag
        denom_dof = n_reg - 2 * lag - 1

        if denom_dof <= 0 or rss_u <= 0:
            continue

        denominator = rss_u / denom_dof
        if denominator <= 0:
            continue

        f_stat = numerator / denominator

        # Compute p-value: P(F > f_stat) with df1=lag, df2=denom_dof
        p_value = stats.f.sf(f_stat, lag, denom_dof)

        # Track the lag with minimum p-value
        if p_value < best_result["p_value"]:
            # Compute coef_sign: sign of sum of x-lag coefficients
            x_lags_coef = fit_u["coef"][lag + 1 : lag + 1 + lag]
            coef_sum = np.sum(x_lags_coef)
            coef_sign = np.sign(coef_sum)

            best_result["f_stat"] = float(f_stat)
            best_result["p_value"] = float(p_value)
            best_result["best_lag"] = lag
            best_result["coef_sign"] = float(coef_sign)

    return best_result


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


# =============================================================================
# GRANGER CAUSALITY PAIRS STRATEGY
# =============================================================================

class GrangerPairsStrategy(Strategy):
    """
    Standalone strategy detecting Granger causality between asset pairs.

    Performs rolling Granger F-test (lags 1-3) between all asset pairs in the
    returns panel. For the current symbol i, tests every other column j as a
    leader. Picks the j with minimum p-value. If p < p_threshold, emits a signal
    on i in the direction implied by yesterday's j-return × the fitted coefficient
    sign, gated on Kronos agreement.

    Context requirements:
        - context["returns_window"]: DataFrame of daily returns (columns=symbols, rows=dates)
        - context["symbol"]: The target symbol being evaluated

    Parameters
    ----------
    p_threshold : float, default 0.05
        P-value threshold for Granger significance.
    max_lag : int, default 3
        Maximum lag to test in Granger F-test (1..max_lag).
    lookback : int, default 120
        Minimum number of rows required in returns_window.
    stop_pct : float, default 15.0
        Stop-loss percentile (for longs; reversed for shorts).
    target_pct : float, default 85.0
        Take-profit percentile (for longs; reversed for shorts).
    """
    name = "granger_pairs"

    def __init__(self, p_threshold: float = 0.05, max_lag: int = 3,
                 lookback: int = 120, stop_pct: float = 15.0,
                 target_pct: float = 85.0):
        self.p_threshold = p_threshold
        self.max_lag = max_lag
        self.lookback = lookback
        self.stop_pct = int(stop_pct)
        self.target_pct = int(target_pct)

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        """
        Test all asset pairs for Granger causality and emit signal if leader found.

        Returns Signal if a significant leader j is found and Kronos agrees with
        the implied direction, else None.
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
            # Extract target return series
            y = returns_window[symbol].values  # Shape (n,)

            if len(y) < self.lookback:
                return None

            # Get all other symbols (potential leaders)
            other_symbols = [s for s in returns_window.columns if s != symbol]
            if not other_symbols:
                return None

            # Test each leader and find the one with minimum p-value
            best_leader = None
            best_p_value = 1.0
            best_coef_sign = 0.0
            best_lag = 1

            for leader_symbol in other_symbols:
                x = returns_window[leader_symbol].values  # Shape (n,)

                # Run Granger F-test
                test_result = granger_f_test(y, x, max_lag=self.max_lag)

                # Track the leader with minimum p-value
                if test_result["p_value"] < best_p_value:
                    best_p_value = test_result["p_value"]
                    best_leader = leader_symbol
                    best_coef_sign = test_result["coef_sign"]
                    best_lag = test_result["best_lag"]

        except Exception:
            return None

        # If best p-value doesn't meet threshold, no signal
        if best_p_value > self.p_threshold or best_leader is None:
            return None

        # Check Kronos agreement
        kronos_mean = dist.stats["close"]["mean"]
        kronos_direction = np.sign(kronos_mean - current_price)

        if kronos_direction == 0:
            return None

        # Compute implied direction: yesterday's leader return × coef_sign
        yesterday_leader_return = returns_window[best_leader].iloc[-2]
        implied_move = yesterday_leader_return * best_coef_sign
        implied_direction = np.sign(implied_move)

        # Check agreement between implied direction and Kronos direction
        if implied_direction != kronos_direction or implied_direction == 0:
            return None

        # Emit signal
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

        # Confidence: 1 - p_value
        confidence = min(max(1.0 - best_p_value, 0.0), 1.0)

        return Signal(
            direction=direction,
            size=size,
            entry=current_price,
            stop=stop,
            target=target,
            strategy_name=self.name,
            confidence=confidence,
            expected_value=ev,
            metadata={
                "leader_symbol": best_leader,
                "p_value": float(best_p_value),
                "best_lag": int(best_lag),
            },
        )


# =============================================================================
# SEASONALITY FILTER
# =============================================================================

def _newey_west_hac_se(resid: np.ndarray, X: np.ndarray, lag: int = 5) -> np.ndarray:
    """
    Compute Newey-West (HAC) standard errors for OLS coefficients.

    Uses Bartlett kernel with the given lag to account for autocorrelation
    in residuals. Returns the standard errors.

    Parameters
    ----------
    resid : np.ndarray, shape (n,)
        OLS residuals.
    X : np.ndarray, shape (n, k)
        Design matrix (should include intercept if needed).
    lag : int, default 5
        Maximum lag for Newey-West weighting.

    Returns
    -------
    np.ndarray, shape (k,)
        HAC-adjusted standard errors.
    """
    n, k = X.shape

    # Compute (X'X)^-1
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        # Return NaN if singular
        return np.full(k, np.nan)

    # Compute Newey-West variance matrix
    # S = sum_{j=-lag}^{lag} w_j * sum_t (ε_t * ε_{t+j} * X_t * X_{t+j}')
    # where w_j = 1 - |j|/(lag+1) is the Bartlett weight

    S = np.zeros((k, k))

    # Lag 0: E[ε_t^2 * X_t * X_t']
    eps_X = resid[:, np.newaxis] * X  # Shape (n, k)
    S += eps_X.T @ eps_X  # Shape (k, k)

    # Positive and negative lags with Bartlett weights
    for j in range(1, lag + 1):
        w_j = 1.0 - j / (lag + 1)

        # Lag j: E[ε_t * ε_{t-j} * X_t * X_{t-j}']
        # When we do element-wise multiply of shifted arrays, we need to align them
        eps_j = resid[:-j]  # ε_{t-j}, shape (n-j,)
        X_j = X[:-j, :]      # X_{t-j}, shape (n-j, k)
        eps_t = resid[j:]    # ε_t, shape (n-j,)
        X_t = X[j:, :]       # X_t, shape (n-j, k)

        # Element-wise product of eps
        eps_prod = (eps_t * eps_j)[:, np.newaxis]  # Shape (n-j, 1)

        # (ε_t * ε_{t-j}) * X_t * X_{t-j}'
        term = eps_prod * X_t  # Shape (n-j, k)
        term_XtXj = term.T @ X_j  # Shape (k, k)

        # Add both lag j and lag -j (by symmetry)
        S += w_j * (term_XtXj + term_XtXj.T)

    # Variance: (X'X)^-1 * S * (X'X)^-1
    cov_hac = XtX_inv @ S @ XtX_inv

    # Standard errors from diagonal
    se = np.sqrt(np.diag(cov_hac))
    return se


class SeasonalityFilterStrategy(Strategy):
    """
    Filter wrapper that vetoes signals fighting significant seasonal effects.

    Estimates day-of-week and month-of-year mean daily return effects over
    a trailing window using OLS. Computes HAC-adjusted (Newey-West) t-statistics
    for each effect. If today is predicted to experience a significant seasonal
    effect that opposes the signal direction, returns None (veto); otherwise
    passes through.

    Parameters
    ----------
    base_strategy : Strategy
        The wrapped strategy to filter.
    lookback : int, default 504
        Number of trailing trading days to use for effect estimation (roughly 2 years).
    t_threshold : float, default 2.0
        T-statistic threshold for significance (two-tailed, ~95% CI).
    """
    name = "seasonality_filter"

    def __init__(self, base_strategy: Strategy, lookback: int = 504,
                 t_threshold: float = 2.0):
        self.base_strategy = base_strategy
        self.lookback = lookback
        self.t_threshold = t_threshold

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        """
        Wrap the base strategy and apply seasonality check.

        Returns None (veto) if a significant seasonal effect opposes the signal.
        Otherwise passes through the base signal.
        """
        # Get the base signal first
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context, **kwargs)
        if base_signal is None:
            return None

        # Extract history and ensure we have dates
        if history is None or len(history) < self.lookback:
            # Not enough history; pass through
            return base_signal

        # Get datetime index or date column
        dates = self._extract_dates(history)
        if dates is None:
            # No dates available; pass through
            return base_signal

        # Extract returns from close prices
        closes = history["close"].values.astype(float)
        if len(closes) < 2:
            return base_signal

        returns = np.diff(np.log(closes))  # Log returns

        if len(returns) != len(dates) - 1:
            # Mismatch; pass through
            return base_signal

        # Use trailing lookback rows
        if len(returns) > self.lookback:
            returns = returns[-self.lookback:]
            dates = dates[-self.lookback - 1:]  # One extra for alignment with returns

        # Compute seasonality effects
        dow_effects, month_effects = self._estimate_effects(returns, dates)

        if dow_effects is None or month_effects is None:
            return base_signal

        # Get today's date (last date + 1 business day)
        today_date = self._next_business_day(dates[-1])
        if today_date is None:
            return base_signal

        # Check day-of-week and month-of-year effects for today
        veto = self._should_veto(today_date, dow_effects, month_effects, base_signal.direction)

        if veto:
            return None

        return base_signal

    def _extract_dates(self, history: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract dates from history DataFrame.

        Handles both DatetimeIndex and a date column.

        Returns
        -------
        np.ndarray of pd.Timestamp or None
            Array of dates, or None if extraction fails.
        """
        try:
            # Try DatetimeIndex first
            if isinstance(history.index, pd.DatetimeIndex):
                return history.index.to_numpy()

            # Try a "date" column
            if "date" in history.columns:
                dates = pd.to_datetime(history["date"]).values
                return dates

            # Try to use any DatetimeIndex (even if not the primary index)
            # For now, we only support the above two cases
            return None
        except Exception:
            return None

    def _next_business_day(self, date: pd.Timestamp) -> Optional[pd.Timestamp]:
        """
        Return the next business day after `date`.

        For now, a simple heuristic: add 1 day, skip weekends (5=Sat, 6=Sun).

        Returns
        -------
        pd.Timestamp or None
        """
        try:
            next_date = pd.Timestamp(date) + pd.Timedelta(days=1)

            # Skip weekends
            while next_date.weekday() in [5, 6]:
                next_date += pd.Timedelta(days=1)

            return next_date
        except Exception:
            return None

    def _estimate_effects(self, returns: np.ndarray, dates: np.ndarray) \
            -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Estimate day-of-week and month-of-year mean daily return effects.

        Runs OLS: return_t = c + sum_d (dow_d * I{dow=d}) + sum_m (month_m * I{month=m}) + e_t

        where I{...} are indicator variables.

        Returns
        -------
        (dow_effects, month_effects) where each is a dict {effect_name: t_stat}
            or (None, None) if estimation fails.
        """
        try:
            n = len(returns)
            if n < 20:
                return None, None

            # Convert dates to pandas Timestamps for weekday/month extraction
            dates_ts = pd.to_datetime(dates)

            # returns[i] = log(close[i+1]/close[i]) is the return realized
            # ON dates[i+1], so label each return by its realization date.
            dow_vals = dates_ts[1:].weekday.values
            month_vals = dates_ts[1:].month.values

            # Estimate each category effect via a contrast regression:
            #   return_t = a + b * I{category} + e_t
            # b = (mean return in category) - (mean return outside category),
            # with a Newey-West HAC standard error. Estimating each category
            # directly (rather than one reference-category dummy regression)
            # ensures every day/month gets its own effect and t-stat.
            dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"]
            dow_effects = {}
            for d in range(7):
                eff = self._contrast_effect(returns, dow_vals == d)
                if eff is not None:
                    dow_effects[dow_names[d]] = eff

            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            month_effects = {}
            for m in range(1, 13):
                eff = self._contrast_effect(returns, month_vals == m)
                if eff is not None:
                    month_effects[month_names[m - 1]] = eff

            return dow_effects, month_effects

        except Exception:
            return None, None

    def _contrast_effect(self, returns: np.ndarray, mask: np.ndarray) -> Optional[Dict]:
        """
        Estimate a single category's mean-return effect with a HAC t-stat.

        Regresses returns on [intercept, category dummy]; the dummy coefficient
        is the category mean minus the non-category mean. The standard error is
        Newey-West (Bartlett kernel, lag 5), computed in numpy.

        Returns
        -------
        dict {"coef": float, "tstat": float} or None if the category is
        degenerate (never occurs, or occurs on nearly every day).
        """
        n = len(returns)
        count = int(mask.sum())
        if count < 2 or count > n - 2:
            return None

        X = np.column_stack([np.ones(n), mask.astype(float)])
        fit = _lagged_ols(returns, X)
        se_hac = _newey_west_hac_se(fit["resid"], X, lag=5)
        if not np.all(np.isfinite(se_hac)):
            return None

        coef = float(fit["coef"][1])
        tstat = coef / (float(se_hac[1]) + 1e-10)
        return {"coef": coef, "tstat": float(tstat)}

    def _should_veto(self, today_date: pd.Timestamp, dow_effects: Dict,
                     month_effects: Dict, signal_direction: Direction) -> bool:
        """
        Check if today's seasonal effects oppose the signal direction significantly.

        Returns True (veto) if:
        - A significant (|t| > threshold) DOW or month effect exists for today
        - AND the effect's sign opposes the signal direction

        Returns
        -------
        bool
            True if veto, False otherwise.
        """
        try:
            # Get today's day-of-week and month
            dow_idx = today_date.weekday()  # 0=Mon, ..., 6=Sun
            month_idx = today_date.month - 1  # 0-11

            dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            # Determine signal direction sign
            signal_sign = 1 if signal_direction == Direction.LONG else -1 if signal_direction == Direction.SHORT else 0

            if signal_sign == 0:
                return False

            # Check day-of-week effect (every day has its own contrast estimate)
            dow_name = dow_names[dow_idx]
            if dow_name in dow_effects:
                dow_info = dow_effects[dow_name]
                if abs(dow_info["tstat"]) > self.t_threshold:
                    effect_sign = np.sign(dow_info["coef"])
                    # Veto if effect sign opposes signal direction
                    if effect_sign != 0 and effect_sign != signal_sign:
                        return True

            # Check month-of-year effect
            month_name = month_names[month_idx]
            if month_name in month_effects:
                month_info = month_effects[month_name]
                if abs(month_info["tstat"]) > self.t_threshold:
                    effect_sign = np.sign(month_info["coef"])
                    # Veto if effect sign opposes signal direction
                    if effect_sign != 0 and effect_sign != signal_sign:
                        return True

            return False

        except Exception:
            return False


# =============================================================================
# BAYESIAN ONLINE CHANGEPOINT DETECTION
# =============================================================================

class ChangepointGuardStrategy(Strategy):
    """
    Stateful filter wrapper: Bayesian online changepoint detection on daily returns.

    Implements Adams & MacKay (2007) algorithm with Normal-Inverse-Gamma conjugate
    prior on log returns. Maintains a run-length posterior; when P(run_length < short_run_len)
    exceeds prob_threshold (indicating a regime break), vetoes all signals and starts
    a cooloff counter for cooloff_days.

    Parameters
    ----------
    base_strategy : Strategy
        The wrapped strategy to filter.
    hazard : float, default 1/60
        Hazard rate (probability of changepoint per day). Default expects ~60 trading days
        between regimes.
    cooloff_days : int, default 3
        Number of trading days to veto signals after detecting a regime break.
    short_run_len : int, default 5
        Threshold for "young" run length; high P(r < short_run_len) indicates fresh break.
    prob_threshold : float, default 0.5
        Probability threshold: if P(run_length < short_run_len) > threshold, trigger veto.
    min_history : int, default 30
        Minimum number of observations required before changepoint detection activates.

    Stateful:
        - Maintains run_length posterior (self.run_length_posterior)
        - Maintains sufficient statistics for Normal-Inverse-Gamma (per run length)
        - Maintains cooloff counter (self.cooloff_counter)
        - Must call reset() to clear state before walk-forward folds
    """
    name = "changepoint_guard"

    def __init__(self, base_strategy: Strategy, hazard: float = 1/60,
                 cooloff_days: int = 3, short_run_len: int = 5,
                 prob_threshold: float = 0.5, min_history: int = 30):
        self.base_strategy = base_strategy
        self.hazard = hazard
        self.cooloff_days = cooloff_days
        self.short_run_len = short_run_len
        self.prob_threshold = prob_threshold
        self.min_history = min_history

        # Stateful: run-length posterior and statistics
        self.run_length_posterior = np.array([1.0])  # P(r=0) initially
        self.stats_by_runlength = [self._init_nig_stats()]  # For r=0
        self.cooloff_counter = 0  # Cooloff countdown
        self.n_obs = 0  # Number of observations seen
        self.prev_prob_young = 1.0  # Track previous young probability for change detection

    def reset(self):
        """
        Clear state for a new walk-forward fold.

        Resets run-length posterior, statistics, and cooloff counter.
        """
        self.run_length_posterior = np.array([1.0])
        self.stats_by_runlength = [self._init_nig_stats()]
        self.cooloff_counter = 0
        self.n_obs = 0
        self.prev_prob_young = 1.0

    def _init_nig_stats(self) -> Dict[str, Any]:
        """
        Initialize Normal-Inverse-Gamma sufficient statistics.

        Priors are scaled to DAILY LOG RETURNS (magnitude ~1%): μ₀=0, κ₀=1,
        α₀=1, β₀=1e-4. The prior predictive std is then
        sqrt(β(κ+1)/(ακ)) ≈ 1.4%, matching typical daily volatility.

        IMPORTANT: β₀ must be on the return scale. A generic weak prior
        (β₀=1) implies a prior predictive std of ~140%, which makes the
        fresh-run (changepoint) hypothesis catastrophically unlikely versus
        any fitted run — the detector would never fire.

        Returns
        -------
        dict
            Keys: "kappa", "mu", "alpha", "beta" (sufficient statistics)
        """
        return {
            "kappa": 1.0,     # Precision of prior on mean
            "mu": 0.0,        # Prior mean
            "alpha": 1.0,     # Shape parameter of IG on variance
            "beta": 1e-4,     # Rate parameter of IG on variance (daily-return scale)
        }

    def _update_nig_stats(self, stats: Dict[str, Any], x: float) -> Dict[str, Any]:
        """
        Update Normal-Inverse-Gamma statistics with a new observation.

        Uses the standard Bayesian update for Normal-Inverse-Gamma conjugate prior
        with one new observation x.

        Parameters
        ----------
        stats : dict
            Current sufficient statistics (kappa, mu, alpha, beta).
        x : float
            New observation (log return).

        Returns
        -------
        dict
            Updated statistics.
        """
        kappa_0 = stats["kappa"]
        mu_0 = stats["mu"]
        alpha_0 = stats["alpha"]
        beta_0 = stats["beta"]

        # Posterior update
        kappa_1 = kappa_0 + 1.0
        mu_1 = (kappa_0 * mu_0 + x) / kappa_1
        alpha_1 = alpha_0 + 0.5

        # Sum of squared deviations
        ssd = ((x - mu_0) ** 2 * kappa_0) / kappa_1
        beta_1 = beta_0 + 0.5 * ssd

        return {
            "kappa": kappa_1,
            "mu": mu_1,
            "alpha": alpha_1,
            "beta": beta_1,
        }

    def _predictive_likelihood(self, x: float, stats: Dict[str, Any]) -> float:
        """
        Compute predictive likelihood of observation under Normal-Inverse-Gamma.

        The predictive distribution is Student-t with:
        - Location: mu
        - Scale: sqrt(beta * (kappa + 1) / (alpha * kappa))
        - Degrees of freedom: 2 * alpha

        Parameters
        ----------
        x : float
            Observation (log return).
        stats : dict
            Sufficient statistics (kappa, mu, alpha, beta).

        Returns
        -------
        float
            Log likelihood (or a very negative number if variance is invalid).
        """
        kappa = stats["kappa"]
        mu = stats["mu"]
        alpha = stats["alpha"]
        beta = stats["beta"]

        # Variance of Student-t
        # Var = beta * (kappa + 1) / (alpha * kappa)
        var_numerator = beta * (kappa + 1)
        var_denominator = alpha * kappa

        if var_denominator <= 0 or var_numerator <= 0:
            return -np.inf

        var_t = var_numerator / var_denominator
        scale_t = np.sqrt(var_t)

        if scale_t <= 0 or not np.isfinite(scale_t):
            return -np.inf

        # Student-t PDF: Gamma((df+1)/2) / (Gamma(df/2) * sqrt(df*π*var)) * (1 + (x-mu)^2/(df*var))^(-(df+1)/2)
        # Simplify using log-scale
        df = 2 * alpha
        centered = (x - mu) ** 2

        # Log likelihood (proportional)
        # log p(x) ≈ -0.5 * (df+1) * log(1 + centered / (df * var))
        # But for numerical stability, use a different formulation
        # The Student-t log pdf is:
        # log p(x) = log Gamma((df+1)/2) - log Gamma(df/2) - 0.5*log(df*π*var) - (df+1)/2*log(1 + (x-mu)^2/(df*var))

        try:
            # Use scipy for gamma function
            from scipy.special import loggamma
            term1 = loggamma((df + 1) / 2.0)
            term2 = loggamma(df / 2.0)
            term3 = 0.5 * np.log(df * np.pi * var_t)
            term4 = (df + 1) / 2.0 * np.log(1.0 + centered / (df * var_t))

            ll = term1 - term2 - term3 - term4
            return float(ll)
        except Exception:
            return -np.inf

    def generate_signal(self, dist: KairosDistribution, current_price: float,
                        history: pd.DataFrame, context: Dict, **kwargs) -> Optional[Signal]:
        """
        Wrap base strategy and apply changepoint detection veto.

        Updates run-length posterior on new log return; if P(r < short_run_len) > threshold,
        starts cooloff veto. During cooloff, all signals are vetoed. If base is None,
        passes through None.

        Returns
        -------
        Signal or None
            The base signal (possibly vetoed), or None.
        """
        # Get base signal first
        base_signal = self.base_strategy.generate_signal(dist, current_price, history, context, **kwargs)
        if base_signal is None:
            return None

        # Check if we have enough history
        if history is None or len(history) < self.min_history:
            # Pass through; not enough data for changepoint detection
            return base_signal

        # Extract the latest log return
        closes = history["close"].values.astype(float)
        if len(closes) < 2:
            return base_signal

        log_return = np.log(closes[-1] / closes[-2])

        # Decrement cooloff counter if active
        if self.cooloff_counter > 0:
            self.cooloff_counter -= 1
            # Veto during cooloff
            return None

        # Update run-length posterior with the new observation
        self._update_runlength_posterior(log_return)

        # Check if regime break is detected
        if self._is_regime_break_detected():
            # Start cooloff and veto this signal
            self.cooloff_counter = self.cooloff_days - 1  # -1 because we decrement at next call
            return None

        # No veto; pass through base signal
        return base_signal

    def _update_runlength_posterior(self, x: float):
        """
        Bayesian online changepoint update: given new observation x, update run-length posterior.

        Adams & MacKay algorithm:
        1. For each current run length r, compute predictive likelihood p(x | r)
        2. Update: P(r_t | y_{1:t}) ∝ P(y_t | r_{t-1}) * P(r_t | r_{t-1}, data)
           where r_t ∈ {0, 1, ..., R} (or {0, r_{t-1}+1})
        3. Normalize

        Parameters
        ----------
        x : float
            Log return.
        """
        R = len(self.run_length_posterior)
        hazard_prob = self.hazard

        # Predictive likelihoods for each run length
        likelihoods = np.zeros(R)
        for r in range(R):
            stats = self.stats_by_runlength[r]
            lik = self._predictive_likelihood(x, stats)
            likelihoods[r] = np.exp(lik) if np.isfinite(lik) else 1e-10

        # Posterior grows: new run lengths are r_t ∈ {0, 1, 2, ..., R}
        # r_t = 0: changepoint happened, fresh start
        # r_t = r+1: no changepoint, previous run length r continues to r+1
        posterior_next = np.zeros(R + 1)

        # r_t = 0: all previous runs end, new one starts
        # P(r_t=0 | data) ∝ P(y_t | data) * hazard = sum_r [P(y_t | r) * P(r_{t-1}) * hazard]
        posterior_next[0] = hazard_prob * np.sum(self.run_length_posterior * likelihoods)

        # r_t = r+1 (r = 0..R-1): previous run length r continues
        # P(r_t = r+1 | data) ∝ P(y_t | r) * P(r_{t-1} = r) * (1 - hazard)
        for r in range(R):
            posterior_next[r + 1] = (1.0 - hazard_prob) * self.run_length_posterior[r] * likelihoods[r]

        # Normalize
        total = np.sum(posterior_next)
        if total > 0:
            posterior_next /= total
        else:
            posterior_next = np.ones(R + 1) / (R + 1)

        # Update statistics: each run length r -> r+1 continues its stats
        # r=0 (new run) gets fresh stats
        stats_next = [self._update_nig_stats(self._init_nig_stats(), x)]
        for r in range(R):
            stats_next.append(self._update_nig_stats(self.stats_by_runlength[r], x))

        # Update state
        self.run_length_posterior = posterior_next
        self.stats_by_runlength = stats_next
        self.n_obs += 1

    def _is_regime_break_detected(self) -> bool:
        """
        Detect regime break using P(run_length < short_run_len) > prob_threshold.

        Strategy:
        - For initialization (max_run_length < short_run_len): no detection
        - For normal operation: flag if prob_young > threshold AND we've accumulated
          enough evidence (max_run_length >= short_run_len or change is large)

        Returns
        -------
        bool
            True if regime break is detected.
        """
        max_run_length = len(self.run_length_posterior) - 1

        # Compute current P(r < short_run_len)
        if max_run_length < self.short_run_len:
            prob_young = np.sum(self.run_length_posterior)  # All are "young"
        else:
            prob_young = np.sum(self.run_length_posterior[:self.short_run_len])

        # For initialization: just track prob, don't flag
        if max_run_length < self.short_run_len - 1:
            self.prev_prob_young = prob_young
            return False

        # Once we have more history, check for significant shift
        # A regime break is indicated by:
        # - prob_young > threshold (young run lengths have high probability)
        # - This represents a shift from the initialization distribution
        #   (where all run lengths are naturally young, so prev_prob was high)
        change_in_prob = prob_young - self.prev_prob_young

        # Detect if: threshold exceeded AND either significant change or stable at high level
        detected = (prob_young > self.prob_threshold) and (max_run_length >= self.short_run_len or change_in_prob > 0.1)

        self.prev_prob_young = prob_young
        return detected
