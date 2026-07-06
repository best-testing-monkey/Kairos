"""
ATR (Average True Range) and GARCH volatility tools and wrapper strategies.

Wilder-smoothed ATR computation and ATRBracketStrategy wrapper for dynamic
bracket adjustment based on volatility. GARCH(1,1) filter wrapper for
volatility regime detection and filtering.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from kairos_backtest import Strategy, Signal, Direction


def atr(history: pd.DataFrame, n: int = 14) -> float:
    """
    Compute Wilder-smoothed ATR (Average True Range) from OHLC history.

    Args:
        history: DataFrame with columns [open, high, low, close, volume].
        n: Period for ATR smoothing (default 14).

    Returns:
        Current ATR value, or NaN if insufficient history.
    """
    if len(history) < n + 1:
        return np.nan

    # Extract OHLC columns
    highs = history["high"].values
    lows = history["low"].values
    closes = history["close"].values

    # Compute true ranges
    trs = []
    for i in range(1, len(history)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        tr = max(high_low, high_close, low_close)
        trs.append(tr)

    trs = np.array(trs)

    # Wilder smoothing: first ATR is simple average of first n TRs
    atr_val = np.mean(trs[:n])

    # Subsequent: (prev_ATR * (n-1) + current_TR) / n
    for i in range(n, len(trs)):
        atr_val = (atr_val * (n - 1) + trs[i]) / n

    return float(atr_val)


def fit_garch(returns: np.ndarray) -> dict:
    """
    Fit GARCH(1,1) model via maximum likelihood estimation (scipy L-BFGS-B).

    Uses variance targeting: omega = longrun_variance * (1 - alpha - beta),
    where longrun_variance is the sample variance of returns.

    Args:
        returns: 1-D array of log returns (assumed demeaned or near zero mean).

    Returns:
        Dictionary with keys:
            - "omega": long-run variance coefficient
            - "alpha": ARCH coefficient (news impact)
            - "beta": GARCH coefficient (persistence)
            - "converged": boolean, True if optimizer converged
            - "sigma_forecast": next-day conditional volatility forecast
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) < 10:
        return {
            "omega": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "converged": False,
            "sigma_forecast": np.sqrt(np.var(returns)) if len(returns) > 0 else 1.0,
        }

    # Demean returns for better numerical stability
    mu = np.mean(returns)
    r_centered = returns - mu
    longrun_var = np.var(r_centered)

    # Avoid division by zero or log(0)
    if longrun_var <= 0:
        longrun_var = 1e-8

    # Initial guess: alpha=0.1, beta=0.8 (standard for financial data)
    alpha_init = 0.1
    beta_init = 0.8
    omega_init = longrun_var * (1 - alpha_init - beta_init)
    if omega_init <= 1e-8:
        omega_init = 1e-8

    def garch_likelihood(params):
        """Negative log-likelihood for GARCH(1,1)."""
        alpha, beta = params
        # Variance targeting: omega = longrun_var * (1 - alpha - beta)
        omega = longrun_var * (1 - alpha - beta)

        # Enforce constraints
        if alpha < 1e-8 or beta < 1e-8 or alpha + beta >= 0.9999:
            return 1e10

        if omega <= 1e-8:
            return 1e10

        # Initialize sigma^2
        sigma2 = np.full(len(r_centered), longrun_var)
        nll = 0.0

        for t in range(1, len(r_centered)):
            # Update conditional variance: sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
            sigma2[t] = omega + alpha * (r_centered[t - 1] ** 2) + beta * sigma2[t - 1]

            # Avoid log(negative) or log(0)
            if sigma2[t] <= 1e-8:
                sigma2[t] = 1e-8

            # Negative log-likelihood contribution: 0.5 * (log(sigma2) + r^2/sigma2)
            nll += 0.5 * (np.log(sigma2[t]) + (r_centered[t] ** 2) / sigma2[t])

        return nll

    # Optimize alpha and beta with L-BFGS-B
    bounds = [(1e-8, 0.5), (1e-8, 0.95)]
    result = minimize(
        garch_likelihood,
        [alpha_init, beta_init],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300},
    )

    if result.success:
        alpha_opt, beta_opt = result.x
        omega_opt = longrun_var * (1 - alpha_opt - beta_opt)
        converged = True
    else:
        # Fallback to initial guess if optimization fails
        alpha_opt = alpha_init
        beta_opt = beta_init
        omega_opt = omega_init
        converged = False

    # Compute next-day variance forecast
    # sigma_T+1^2 = omega + alpha * r_T^2 + beta * sigma_T^2
    # where sigma_T^2 is the last computed conditional variance
    sigma2_last = longrun_var
    for t in range(1, len(r_centered)):
        sigma2_last = omega_opt + alpha_opt * (r_centered[t - 1] ** 2) + beta_opt * sigma2_last
        if sigma2_last <= 1e-8:
            sigma2_last = 1e-8

    sigma_forecast = np.sqrt(sigma2_last)

    return {
        "omega": float(omega_opt),
        "alpha": float(alpha_opt),
        "beta": float(beta_opt),
        "converged": bool(converged),
        "sigma_forecast": float(sigma_forecast),
    }


class ATRBracketStrategy(Strategy):
    """
    Wrapper strategy that dynamically adjusts signal brackets based on volatility.

    Recomputes stop at entry ∓ k_stop*ATR(14) and target at entry ± k_target*ATR(14),
    keeping the tighter of {ATR bracket, original bracket} for both stop and target.
    Direction-consistent: stop below entry for LONG, above for SHORT.

    Args:
        base_strategy: Strategy instance to wrap.
        k_stop: ATR multiplier for stop bracket (default 2.0).
        k_target: ATR multiplier for target bracket (default 3.0).
        n: ATR period (default 14).
    """

    name = "atr_bracket"

    def __init__(self, base_strategy: Strategy, k_stop: float = 2.0,
                 k_target: float = 3.0, n: int = 14):
        self.base_strategy = base_strategy
        self.k_stop = k_stop
        self.k_target = k_target
        self.n = n

    def generate_signal(self, dist, current_price, history, context, **kwargs):
        """
        Generate signal from base strategy and adjust brackets by ATR.

        Returns:
            Adjusted Signal or None if base returns None.
        """
        # Call base strategy
        base_signal = self.base_strategy.generate_signal(
            dist, current_price, history, context, **kwargs
        )
        if base_signal is None:
            return None

        # Compute ATR
        atr_val = atr(history, n=self.n)
        if np.isnan(atr_val) or atr_val <= 0:
            # If ATR can't be computed, return base signal unchanged
            return base_signal

        entry = base_signal.entry
        direction = base_signal.direction

        # Compute ATR-based brackets
        if direction == Direction.LONG:
            # For LONG: stop below entry, target above entry
            atr_stop = entry - self.k_stop * atr_val
            atr_target = entry + self.k_target * atr_val

            # Keep tighter stop (higher stop for LONG)
            new_stop = max(atr_stop, base_signal.stop)

            # Keep tighter target (lower target for LONG)
            new_target = min(atr_target, base_signal.target)
        else:  # Direction.SHORT
            # For SHORT: stop above entry, target below entry
            atr_stop = entry + self.k_stop * atr_val
            atr_target = entry - self.k_target * atr_val

            # Keep tighter stop (lower stop for SHORT)
            new_stop = min(atr_stop, base_signal.stop)

            # Keep tighter target (higher target for SHORT)
            new_target = max(atr_target, base_signal.target)

        # Return new Signal with adjusted brackets, preserving other fields
        return Signal(
            direction=direction,
            size=base_signal.size,
            entry=entry,
            stop=new_stop,
            target=new_target,
            strategy_name=self.name,
            confidence=base_signal.confidence,
            expected_value=base_signal.expected_value,
            metadata=base_signal.metadata,
        )


class GARCHFilterStrategy(Strategy):
    """
    Wrapper strategy that filters signals based on GARCH(1,1) volatility regimes.

    Fits GARCH(1,1) on trailing log returns, forecasts next-day volatility,
    and blocks the wrapped strategy's signal when forecast volatility exceeds
    a percentile threshold of the trailing fitted volatility series.

    Args:
        base_strategy: Strategy instance to wrap.
        sigma_cap_pct: Percentile threshold for blocking (default 90.0).
                       Block when forecast_sigma > this percentile of trailing sigmas.
        lookback: Window size for GARCH fitting (default 250).
        refit_days: Refit GARCH model every N bars (default 5 = weekly).
    """

    name = "garch_filter"

    def __init__(self, base_strategy: Strategy, sigma_cap_pct: float = 90.0,
                 lookback: int = 250, refit_days: int = 5):
        self.base_strategy = base_strategy
        self.sigma_cap_pct = sigma_cap_pct
        self.lookback = lookback
        self.refit_days = refit_days

        # State tracking
        self._bar_count = 0  # Track bars since last refit
        self._sigma_history = []  # List of fitted conditional volatilities
        self._last_fit = None  # Cached GARCH fit result
        self._converged = True  # Track convergence status

    def reset(self):
        """Reset internal state for walk-forward validation."""
        self._bar_count = 0
        self._sigma_history = []
        self._last_fit = None
        self._converged = True

    def generate_signal(self, dist, current_price, history, context, **kwargs):
        """
        Generate signal from base strategy and filter by GARCH volatility.

        Returns:
            Signal if base strategy generates one and volatility is below cap,
            None otherwise (either base blocked or GARCH blocked).
        """
        # Call base strategy first
        base_signal = self.base_strategy.generate_signal(
            dist, current_price, history, context, **kwargs
        )
        if base_signal is None:
            return None

        # Compute log returns from history
        closes = history["close"].values
        if len(closes) < 2:
            # Not enough data; pass through with warning
            context["garch_warning"] = True
            return base_signal

        log_returns = np.diff(np.log(closes))

        # Refit GARCH if needed
        if self._bar_count % self.refit_days == 0:
            # Use trailing lookback bars for fitting
            fit_window = log_returns[-self.lookback:] if len(log_returns) >= self.lookback else log_returns
            self._last_fit = fit_garch(fit_window)
            self._converged = self._last_fit.get("converged", False)

            # Track this fitted sigma in history
            if self._converged:
                self._sigma_history.append(self._last_fit["sigma_forecast"])

        self._bar_count += 1

        # If GARCH fit did not converge, pass through with warning
        if not self._converged:
            context["garch_warning"] = True
            return base_signal

        # Get forecast volatility and cap threshold
        if self._last_fit is None:
            # No fit available yet; pass through
            context["garch_warning"] = True
            return base_signal

        forecast_sigma = self._last_fit["sigma_forecast"]

        # Compute sigma_cap as percentile of trailing fitted sigmas
        if len(self._sigma_history) < 2:
            # Not enough history; pass through
            return base_signal

        sigma_cap = np.percentile(self._sigma_history, self.sigma_cap_pct)

        # Block signal if forecast sigma exceeds cap
        if forecast_sigma > sigma_cap:
            return None

        return base_signal


class VolTargetSizerStrategy(Strategy):
    """
    Wrapper strategy that scales signal sizes based on blended realized and predicted volatility.

    Sizing wrapper: blended_vol = 0.5*(GARCH forecast) + 0.5*(Kronos predicted range vol).
    Scales signal.size by target_vol / blended_vol, clipped so final size <= base size * max_leverage.
    Never increases a zero-size signal; passes through None.

    Kronos predicted range vol = (pct_84 - pct_16) / (2 * current_price), annualized by sqrt(252).

    GARCH forecast is cached and refit every refit_days bars (weekly by default).

    Args:
        base_strategy: Strategy instance to wrap.
        target_vol: Target annualized volatility (default 0.15 = 15%).
        max_leverage: Maximum leverage cap on final size (default 2.0).
        lookback: Window size for GARCH fitting (default 250).
        refit_days: Refit GARCH model every N bars (default 5 = weekly).
    """

    name = "vol_target_sizer"

    def __init__(self, base_strategy: Strategy, target_vol: float = 0.15,
                 max_leverage: float = 2.0, lookback: int = 250, refit_days: int = 5):
        self.base_strategy = base_strategy
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.lookback = lookback
        self.refit_days = refit_days

        # State tracking
        self._bar_count = 0  # Track bars since last refit
        self._sigma_history = []  # List of fitted conditional volatilities
        self._last_fit = None  # Cached GARCH fit result
        self._converged = True  # Track convergence status

    def reset(self):
        """Reset internal state for walk-forward validation."""
        self._bar_count = 0
        self._sigma_history = []
        self._last_fit = None
        self._converged = True

    def generate_signal(self, dist, current_price, history, context, **kwargs):
        """
        Generate signal from base strategy and scale by vol targeting.

        Returns:
            Scaled Signal or None if base returns None.
        """
        # Call base strategy first
        base_signal = self.base_strategy.generate_signal(
            dist, current_price, history, context, **kwargs
        )
        if base_signal is None:
            return None

        # Never increase a zero-size signal
        if base_signal.size == 0.0:
            return base_signal

        # Compute log returns from history
        closes = history["close"].values
        if len(closes) < 2:
            # Not enough data; return base signal unchanged
            return base_signal

        log_returns = np.diff(np.log(closes))

        # Refit GARCH if needed
        if self._bar_count % self.refit_days == 0:
            # Use trailing lookback bars for fitting
            fit_window = log_returns[-self.lookback:] if len(log_returns) >= self.lookback else log_returns
            self._last_fit = fit_garch(fit_window)
            self._converged = self._last_fit.get("converged", False)

            # Track this fitted sigma in history
            if self._converged:
                self._sigma_history.append(self._last_fit["sigma_forecast"])

        self._bar_count += 1

        # Compute GARCH-based volatility (daily forecast, annualized)
        garch_vol = 0.0
        if self._converged and self._last_fit is not None:
            # fit_garch returns daily sigma; annualize it
            daily_sigma = self._last_fit["sigma_forecast"]
            garch_vol = daily_sigma * np.sqrt(252)
        else:
            # Fallback to realized vol if GARCH fails
            if len(log_returns) >= 20:
                garch_vol = np.std(log_returns[-20:]) * np.sqrt(252)
            else:
                garch_vol = np.std(log_returns) * np.sqrt(252)

        # Compute Kronos predicted range vol
        # range_vol = (pct_84 - pct_16) / (2 * price), annualized by sqrt(252)
        close_stats = dist.stats.get("close", {})
        pct_84 = close_stats.get(f"pct_{int(84)}", current_price)
        pct_16 = close_stats.get(f"pct_{int(16)}", current_price)

        if current_price > 0:
            # Range-based vol: (84th percentile - 16th percentile) / (2 * price)
            kronos_vol_daily = (pct_84 - pct_16) / (2.0 * current_price)
            kronos_vol = kronos_vol_daily * np.sqrt(252)
        else:
            kronos_vol = 0.0

        # Blend: 50% GARCH + 50% Kronos
        blended_vol = 0.5 * garch_vol + 0.5 * kronos_vol

        # Avoid division by zero
        if blended_vol <= 0:
            blended_vol = self.target_vol

        # Scale size by target_vol / blended_vol
        size_multiplier = self.target_vol / blended_vol

        # Clip to max_leverage
        size_multiplier = min(size_multiplier, self.max_leverage)

        # Apply multiplier to base size, but never reduce below 0
        new_size = max(0.0, base_signal.size * size_multiplier)

        # Return scaled Signal
        return Signal(
            direction=base_signal.direction,
            size=new_size,
            entry=base_signal.entry,
            stop=base_signal.stop,
            target=base_signal.target,
            strategy_name=self.name,
            confidence=base_signal.confidence,
            expected_value=base_signal.expected_value,
            metadata=base_signal.metadata,
        )
