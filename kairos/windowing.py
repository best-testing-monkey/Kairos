"""KAI-3: Bar-count to date-range windowing.

Converts (lookback, interval, end_date) into a start_date that yields at
least `lookback` bars from price_cache, with a widen-and-retry loop to
handle holiday clusters.
"""
from __future__ import annotations

import math
from datetime import timedelta

import pandas as pd

from .errors import InsufficientDataError, UnsupportedIntervalError

_BARS_PER_DAY: dict[str, float] = {
    "1m":  390.0,
    "2m":  195.0,
    "5m":   78.0,
    "15m":  26.0,
    "30m":  13.0,
    "60m":   6.5,
    "90m":   4.3,
    "1h":    6.5,
    "1d":    1.0,
    "5d":    0.2,
    "1wk":   0.2,
    "1mo":   0.05,
    "3mo":   0.017,
}

_RETRY_CAP = 4


def estimate_start(
    end_date: pd.Timestamp,
    interval: str,
    lookback: int,
    buffer: float = 2.0,
) -> pd.Timestamp:
    """Return a generous start_date for a price_cache call.

    The estimate over-fetches to account for weekends, holidays, and partial
    sessions; the caller tails the result to exactly *lookback* rows.
    """
    interval = interval.lower().strip()
    if interval not in _BARS_PER_DAY:
        raise UnsupportedIntervalError(
            f"Unsupported interval {interval!r}. "
            f"Supported: {', '.join(sorted(_BARS_PER_DAY))}"
        )
    per_day = _BARS_PER_DAY[interval]
    trading_days_needed = math.ceil(lookback / per_day) * buffer
    calendar_days = trading_days_needed * 7 / 5 + 10  # weekend + holiday slack
    return end_date - timedelta(days=int(math.ceil(calendar_days)))


def fetch_with_retry(
    symbol: str,
    interval: str,
    lookback: int,
    end_date: pd.Timestamp,
    fetch_fn,  # callable(symbol, start_str, end_str, interval) -> pd.DataFrame | None
    retry_cap: int = _RETRY_CAP,
) -> pd.DataFrame:
    """Fetch price data, widening the window until *lookback* bars are available.

    Args:
        symbol: Ticker symbol.
        interval: Bar interval string.
        lookback: Required number of bars.
        end_date: Last bar date (inclusive).
        fetch_fn: Callable with the same signature as price_cache.get_price_data.
        retry_cap: Maximum number of fetch attempts.

    Returns:
        Raw price_cache DataFrame with at least *lookback* rows.

    Raises:
        InsufficientDataError: Still short after *retry_cap* attempts.
    """
    buffer = 2.0
    last_have = 0
    for attempt in range(retry_cap + 1):
        start_date = estimate_start(end_date, interval, lookback, buffer=buffer)
        raw = fetch_fn(
            symbol,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            interval=interval,
        )
        if raw is not None and len(raw) >= lookback:
            return raw
        last_have = len(raw) if raw is not None else 0
        buffer *= 2.0  # double the lookback window each retry

    raise InsufficientDataError(have=last_have, want=lookback)
