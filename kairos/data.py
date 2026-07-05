"""KAI-5: Public orchestration entry point - get_forecast_window.

Assembles price_cache data into the (x_df, x_timestamp, y_timestamp) tuple
expected by KronosPredictor.predict.
"""
from __future__ import annotations

import pandas as pd

import price_cache

from .adapter import to_kronos_frame
from .calendars import future_timestamps
from .config import _state
from .errors import NoDataError, UnsupportedIntervalError
from .windowing import fetch_with_retry

_SUPPORTED_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo",
}


def get_forecast_window(
    symbol: str,
    interval: str,
    lookback: int,
    pred_len: int,
    *,
    end: pd.Timestamp | str | None = None,
    amount: str = "omit",
    calendar: str | None = None,
    tz: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Fetch and shape price data into KronosPredictor.predict inputs.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        interval: Bar interval string accepted by price_cache.
        lookback: Number of historical bars required (== len(x_df)).
        pred_len: Number of future bars to predict (== len(y_timestamp)).
        end: Last bar date.  Defaults to now.
        amount: "omit" | "auto" | "close_volume" - see adapter.py.
        calendar: exchange_calendars code; defaults to configured value.
        tz: IANA timezone; defaults to configured value.

    Returns:
        (x_df, x_timestamp, y_timestamp) ready for KronosPredictor.predict.

    Raises:
        UnsupportedIntervalError: *interval* not in price_cache's set.
        NoDataError: price_cache returned None.
        InsufficientDataError: Fewer than *lookback* bars after retries.
        DataQualityError: NaN in OHLCV columns.
        CalendarError: Future timestamp generation failed.
    """
    _validate_interval(interval)

    eff_tz = tz or _state.tz
    eff_calendar = calendar or _state.calendar

    if end is None:
        # Live mode: normalize to today. price_cache never caches or delivers
        # incomplete (in-progress) bars, so no partial-bar dropping is needed.
        end_date = pd.Timestamp.now(tz=eff_tz).normalize()
    elif isinstance(end, str):
        end_date = pd.Timestamp(end, tz=eff_tz).normalize()
    else:
        end_date = pd.Timestamp(end).normalize()
        if end_date.tzinfo is None:
            end_date = end_date.tz_localize(eff_tz)

    _got_none = False
    _db_path = price_cache.DB_PATH  # read at call time, not module-load time

    def _fetch(sym, start_str, end_str, interval):
        nonlocal _got_none
        raw = price_cache.get_price_data(sym, start_str, end_str, interval=interval,
                                         db_path=_db_path)
        if raw is None:
            _got_none = True
        return raw

    from .errors import InsufficientDataError as _ISE
    try:
        raw = fetch_with_retry(symbol, interval, lookback, end_date, _fetch)
    except _ISE:
        if _got_none:
            raise NoDataError(symbol, "?", end_date.strftime("%Y-%m-%d"), interval)
        raise

    x_df, x_timestamp = to_kronos_frame(raw, lookback, amount=amount)
    y_timestamp = future_timestamps(
        x_timestamp.iloc[-1], interval, pred_len, eff_calendar, eff_tz
    )
    return x_df, x_timestamp, y_timestamp


def _validate_interval(interval: str) -> None:
    if interval.lower().strip() not in _SUPPORTED_INTERVALS:
        raise UnsupportedIntervalError(
            f"Unsupported interval {interval!r}. "
            f"Supported: {', '.join(sorted(_SUPPORTED_INTERVALS))}"
        )
