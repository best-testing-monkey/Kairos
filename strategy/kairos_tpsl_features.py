"""E18-S02: Price-history-derived feature extraction for TP/SL optimization.

Extracts ML model input features from historical OHLCV bars, entirely from
re-fetchable price history (no GPU, no distribution reconstruction). Features
are categorical (asset_class, interval) or numeric (ATR, volatility, trend, etc.).

No-lookahead guarantee: output is unchanged whether history is exactly truncated
at `as_of` or has extra rows appended after it.
"""

from typing import Any

import numpy as np
import pandas as pd
from kairos_strategies import asset_class_for


def extract_features(
    ticker: str,
    as_of: str,
    interval: str,
    entry: float,
    history: pd.DataFrame,
) -> dict[str, Any]:
    """Extract ML model features from historical price data.

    Args:
        ticker: Single ticker symbol (e.g., "AAPL").
        as_of: ISO date string (e.g., "2024-01-15"), used to truncate history
               to ensure no lookahead -- output is identical whether history is
               exactly truncated at as_of or has extra rows appended after.
        interval: Bar interval as string (e.g., "1d", "1h").
        entry: Entry price for position-in-range calculation.
        history: DataFrame with columns [open, high, low, close, volume],
                 indexed by datetime. Will be truncated to as_of internally.

    Returns:
        dict[str, Any] with keys:
            - "atr": ATR(14) in price units
            - "realized_vol": 20-bar rolling stdev of returns
            - "asset_class": asset class name (str)
            - "interval": bar interval string (str)
            - "trend_10": 10-bar momentum: (close[-1] - close[-10]) / close[-10]
            - "range_position": position within 20-bar high-low range

    Raises:
        ValueError: if history has fewer bars than required for largest lookback window.
    """
    # Truncate history to as_of to ensure no lookahead (no-lookahead guarantee)
    # Normalize as_of's tz-awareness to match history.index to avoid comparison crash
    as_of_date = pd.Timestamp(as_of)
    if isinstance(history.index, pd.DatetimeIndex) and history.index.tz is not None:
        # history.index is tz-aware: localize/convert as_of to match
        if as_of_date.tz is None:
            # Assume UTC if as_of is naive; convert to match history's tz
            as_of_date = as_of_date.tz_localize("UTC").tz_convert(history.index.tz)
        else:
            # as_of is already aware; convert to match history's tz
            as_of_date = as_of_date.tz_convert(history.index.tz)
    # else: both are naive, comparison is safe
    history_truncated = history[history.index <= as_of_date]

    if len(history_truncated) < 21:  # Need 21 bars: 20 for realized_vol + 1 base
        raise ValueError(
            f"Insufficient history: need >=21 bars, got {len(history_truncated)}"
        )

    closes = np.array(history_truncated["close"].values, dtype=float)
    highs = np.array(history_truncated["high"].values, dtype=float)
    lows = np.array(history_truncated["low"].values, dtype=float)

    # ===== ATR(14) =====
    atr_val = _compute_atr(history_truncated, n=14)

    # ===== Realized volatility (20-bar rolling stdev of returns) =====
    log_returns = np.diff(np.log(closes))
    realized_vol = float(np.std(log_returns[-20:], ddof=1))

    # ===== Asset class =====
    asset_class = asset_class_for([ticker])

    # ===== Interval (pass-through) =====
    interval_val = interval

    # ===== Trend/momentum (10-bar) =====
    # (close[-1] - close[-11]) / close[-11], where close[-11] is 10 bars back
    if len(closes) < 11:
        raise ValueError(
            f"Insufficient history for trend: need >=11 bars, got {len(history)}"
        )
    trend_10 = (closes[-1] - closes[-11]) / closes[-11]

    # ===== Position within 20-bar range =====
    # (close[-1] - low[-20:]) / (high[-20:] - low[-20:])
    # Guard against zero denominator: if high==low (flat/halted instrument), return 0.5
    recent_high = np.max(highs[-20:])
    recent_low = np.min(lows[-20:])
    range_diff = recent_high - recent_low
    if range_diff < 1e-9:  # Effectively zero (covers floating-point precision)
        range_position = 0.5  # Sentinel: middle of degenerate range
    else:
        range_position = (closes[-1] - recent_low) / range_diff

    return {
        "atr": float(atr_val),
        "realized_vol": realized_vol,
        "asset_class": asset_class,
        "interval": interval_val,
        "trend_10": float(trend_10),
        "range_position": float(range_position),
    }


def _compute_atr(history: pd.DataFrame, n: int = 14) -> float:
    """Compute Wilder-smoothed ATR from OHLC history.

    Reuses the exact formula from ATRBracketStrategy in kairos_volatility.py.
    Returns NaN if insufficient history.

    Args:
        history: DataFrame with columns [open, high, low, close, volume].
        n: Period for ATR smoothing (default 14).

    Returns:
        Current ATR value, or NaN if len(history) < n + 1.
    """
    if len(history) < n + 1:
        return np.nan

    # Extract OHLC columns
    highs = np.array(history["high"].values, dtype=float)
    lows = np.array(history["low"].values, dtype=float)
    closes = np.array(history["close"].values, dtype=float)

    # Compute true ranges
    trs_list: list[float] = []
    for i in range(1, len(history)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        tr = max(high_low, high_close, low_close)
        trs_list.append(tr)

    trs = np.array(trs_list, dtype=float)

    # Wilder smoothing: first ATR is simple average of first n TRs
    atr_val = np.mean(trs[:n])

    # Subsequent: (prev_ATR * (n-1) + current_TR) / n
    for i in range(n, len(trs)):
        atr_val = (atr_val * (n - 1) + trs[i]) / n

    return float(atr_val)
