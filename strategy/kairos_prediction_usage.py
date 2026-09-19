"""
kairos_prediction_usage.py
===========================
Prediction-usage modes: transformations of AssetPrediction inputs before
they reach strategy logic.

Contains helpers and mode functions for different ways to present a
KairosDistribution forecast to indicator/strategy code. Modes are wired
into the dispatcher in kairos_orchestrator.py via the PREDICTION_USAGE_MODES
registry.

Usage:
    from kairos_prediction_usage import _build_synthetic_bar

    pred = AssetPrediction(symbol="BTC-USD", dist=dist_obj, ...)
    synth_bar = _build_synthetic_bar(pred, interval="1d")
"""

import pandas as pd
import re
from datetime import timedelta


def _interval_to_timedelta(interval: str) -> timedelta:
    """Convert an interval string (e.g., "1d", "1h", "60m", "30m", "1wk")
    to a fixed timedelta bar size. Calendar-based units ("1mo", "3mo")
    have no fixed duration and are not supported.

    Args:
        interval: Interval string in the format "{count}{unit}" where unit is
                 one of: m (minute), h (hour), d (day), wk (week).
                 Examples: "1h", "4h", "1d", "1wk", "60m".

    Returns:
        A timedelta object representing the interval duration.

    Raises:
        ValueError: If the interval format is invalid or uses calendar-based
                   units ("1mo", "3mo").
    """
    match = re.fullmatch(r"(\d+)(mo|wk|d|h|m)", interval)
    if not match or match.group(2) == "mo":
        raise ValueError(f"Cannot convert interval {interval!r} to a fixed timedelta step")
    count = int(match.group(1))
    unit = match.group(2)

    unit_mapping = {
        "m": lambda n: timedelta(minutes=n),
        "h": lambda n: timedelta(hours=n),
        "d": lambda n: timedelta(days=n),
        "wk": lambda n: timedelta(weeks=n),
    }
    return unit_mapping[unit](count)


def _build_synthetic_bar(
    pred, interval: str
) -> pd.Series:
    """Build one synthetic OHLCV bar from a KairosDistribution's median
    percentile stats and the current price.

    The synthetic bar represents the model's forecast as a single bar that can
    be appended to historical OHLCV data. This allows indicator-based strategies
    (RSI, MACD, etc.) to incorporate the forecast into their calculations
    without strategy-level changes. Volume is carried forward from the last
    real bar unchanged (v1 simplification).

    Args:
        pred: AssetPrediction object containing:
            - dist: KairosDistribution with .stats["open"/"high"/"low"/"close"]
                   dicts, each containing percentile keys like "pct_50"
            - current_price: float, the entry price anchor
            - history: pd.DataFrame with OHLCV data; last row used for volume
                      and timestamp advancement
        interval: str, interval string (e.g., "1d", "1h") used to calculate
                 the next bar's timestamp

    Returns:
        A pd.Series with columns ["open", "high", "low", "close", "volume"],
        indexed by (named with) the next timestamp one interval step after
        pred.history.index[-1].

    Example:
        >>> pred = AssetPrediction(
        ...     symbol="BTC-USD",
        ...     dist=dist_obj,  # has .stats with median percentiles
        ...     current_price=50000.0,
        ...     history=df  # last row is '2026-01-01 00:00:00'
        ... )
        >>> bar = _build_synthetic_bar(pred, interval="1d")
        >>> bar.name  # '2026-01-02 00:00:00'
        >>> bar["open"]  # 50000.0
        >>> bar["high"]  # dist.stats["high"]["pct_50"]
    """
    s = pred.dist.stats
    last_ts = pred.history.index[-1]
    step = _interval_to_timedelta(interval)
    next_ts = last_ts + step

    return pd.Series(
        {
            "open": pred.current_price,
            "high": s["high"]["pct_50"],
            "low": s["low"]["pct_50"],
            "close": s["close"]["pct_50"],
            "volume": pred.history.iloc[-1]["volume"],
        },
        name=next_ts,
    )
