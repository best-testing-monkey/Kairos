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
            # price_cache marks a whole ticker as no-data in its no_data_tickers
            # table after a single failed fetch (e.g. a keyless provider), which
            # can hide data that is already sitting in the local prices table --
            # the same gap that made strategy/kairos_strategies.py's fetch_data_raw
            # crash a finetune_next backtest for a symbol whose training fetch had
            # just succeeded via this exact fallback moments earlier. This is the
            # live signal-generation path, so the same guard applies here too.
            # Only triggered on None (not a merely-empty-but-real DataFrame,
            # which fetch_with_retry already handles by widening the window),
            # mirroring kairos/cli/finetune.py's fallback trigger exactly.
            raw = fetch_price_data_local_fallback(
                sym, pd.Timestamp(start_str), pd.Timestamp(end_str), interval, _db_path,
            )
            if raw is not None and not raw.empty:
                print(f"  [{sym}] price_cache returned None; using direct local SQLite fallback")
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


_INTERVAL_MINUTES = {
    "1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30,
    "60m": 60, "90m": 90, "1h": 60, "1d": 1440,
    "5d": 7200, "1wk": 10080, "1mo": 43200, "3mo": 129600,
}


def fetch_price_data_local_fallback(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
    db_path: str,
) -> pd.DataFrame | None:
    """Fallback read from the local SQLite prices table if get_price_data returns None.

    price_cache marks a whole ticker as no-data in the no_data_tickers table when a
    single fetch fails, which can hide data that is already in the prices table.
    This fallback reads the local table directly, bypassing that guard. Shared by
    kairos/cli/finetune.py (training data) and strategy/kairos_strategies.py's
    fetch_data_raw (backtest data) -- both need it because a ticker can carry a
    stale no-data marker from one context (e.g. a keyless-provider failure) while
    the local prices table already has perfectly good history for it, as seen
    when a finetune_next backtest step crashed on a symbol whose training fetch
    had just succeeded via this exact fallback moments earlier in the same run.
    """
    import sqlite3
    from pathlib import Path

    interval_minutes = _INTERVAL_MINUTES.get(interval.lower())
    if interval_minutes is None:
        raise ValueError(f"Unsupported interval {interval!r}")

    path = Path(db_path).resolve()
    if not path.exists():
        return None

    conn = sqlite3.connect(path, check_same_thread=False)
    try:
        rows = conn.execute(
            """SELECT date, open, high, low, close, volume, dividends, stock_splits, market_cap
               FROM prices
               WHERE ticker=? AND date >= ? AND date <= ? AND interval_minutes=?
               ORDER BY date""",
            (symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), interval_minutes),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    df = pd.DataFrame(
        rows,
        columns=["Date", "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "market_cap"],
    )
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(
        "America/New_York", ambiguous="infer", nonexistent="shift_forward"
    )
    df.set_index("Date", inplace=True)
    return df
