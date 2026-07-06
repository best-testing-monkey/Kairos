"""
ATR (Average True Range) volatility tools and wrapper strategies.

Wilder-smoothed ATR computation and ATRBracketStrategy wrapper for dynamic
bracket adjustment based on volatility.
"""

import numpy as np
import pandas as pd
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
