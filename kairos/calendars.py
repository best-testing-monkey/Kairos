"""KAI-4: Exchange-calendar-aware future timestamp synthesis.

Generates the next pred_len trading timestamps after the last observed bar,
respecting holidays, half-days, DST, and intraday session boundaries.
"""
from __future__ import annotations

import pandas as pd

from .errors import CalendarError

_DAILY_OR_COARSER = {"1d", "5d", "1wk", "1mo", "3mo"}

_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1,
    "2m": 2,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "90m": 90,
    "1h": 60,
}


def _step(interval: str) -> int:
    """Return the number of one-minute bars that make up one interval bar."""
    return _INTERVAL_MINUTES.get(interval.lower(), 1)


def _next_trading_minute(cal, last_ts: pd.Timestamp, interval: str) -> pd.Timestamp:
    """Return the first trading minute strictly after last_ts + interval step."""
    step = pd.Timedelta(minutes=_step(interval))
    candidate = last_ts + step
    # If candidate falls outside a session, advance to the next session open.
    try:
        minutes = cal.minutes_window(candidate, 1)
        if len(minutes) == 0:
            raise CalendarError("No trading minutes after last bar")
        if minutes[0] < candidate:
            # candidate is not a trading minute; find the next open
            sessions = cal.sessions_window(
                cal.minute_to_session(last_ts, direction="previous") + pd.Timedelta(days=1),
                10,
            )
            if len(sessions) == 0:
                raise CalendarError("No future sessions after last bar")
            return cal.session_open(sessions[0]).tz_convert(last_ts.tzinfo)
        return minutes[0]
    except Exception as exc:
        if isinstance(exc, CalendarError):
            raise
        raise CalendarError(f"Failed to compute next trading minute: {exc}") from exc


def future_timestamps(
    last_ts: pd.Timestamp,
    interval: str,
    pred_len: int,
    calendar: str = "XNYS",
    tz: str = "America/New_York",
) -> pd.Series:
    """Return the next *pred_len* trading timestamps after *last_ts*.

    Args:
        last_ts: Timestamp of the last observed bar (from x_timestamp).
        interval: Bar interval string (e.g. "1d", "5m").
        pred_len: Number of future bars to generate.
        calendar: exchange_calendars calendar code.
        tz: IANA timezone name for the returned timestamps.

    Returns:
        pd.Series named "y_timestamp", length pred_len, tz-aware, strictly
        increasing, with y_timestamp[0] > last_ts.

    Raises:
        CalendarError: On any calendar resolution failure.
    """
    try:
        import exchange_calendars as xcals
    except ImportError as exc:
        raise CalendarError("exchange_calendars is required for timestamp synthesis") from exc

    try:
        cal = xcals.get_calendar(calendar)
    except Exception as exc:
        raise CalendarError(f"Cannot load calendar {calendar!r}: {exc}") from exc

    interval = interval.lower().strip()

    try:
        if interval in _DAILY_OR_COARSER:
            future = _daily_future(cal, last_ts, interval, pred_len, tz)
        else:
            future = _intraday_future(cal, last_ts, interval, pred_len, tz)
    except CalendarError:
        raise
    except Exception as exc:
        raise CalendarError(f"Timestamp generation failed: {exc}") from exc

    if len(future) != pred_len:
        raise CalendarError(
            f"Generated {len(future)} timestamps but need {pred_len}"
        )
    return pd.Series(future, name="y_timestamp")


def _daily_future(cal, last_ts, interval, pred_len, tz) -> pd.DatetimeIndex:
    """Generate daily/weekly/monthly future session timestamps."""
    step = {"1d": 1, "5d": 5, "1wk": 1, "1mo": 1, "3mo": 1}.get(interval, 1)

    # Find the first session strictly after last_ts
    last_date = pd.Timestamp(last_ts).normalize().date()
    session_start = cal.date_to_session(
        pd.Timestamp(last_date) + pd.Timedelta(days=1), direction="next"
    )

    # Gather enough sessions (step skips for 5d; weekly/monthly handled as 1 session)
    n_sessions = pred_len * step
    sessions = cal.sessions_window(session_start, n_sessions)

    if interval == "5d":
        selected = sessions[::5][:pred_len]
    else:
        selected = sessions[:pred_len]

    return pd.DatetimeIndex(selected).tz_localize(tz)


def _next_session(cal, after_session: pd.Timestamp) -> pd.Timestamp | None:
    """Return the next valid calendar session after *after_session*."""
    candidate = after_session + pd.Timedelta(days=1)
    end_bound = after_session + pd.Timedelta(days=30)
    while candidate <= end_bound:
        try:
            sess = cal.date_to_session(candidate, direction="next")
            if sess > after_session:
                return sess
        except Exception:
            pass
        candidate += pd.Timedelta(days=1)
    return None


def _intraday_future(cal, last_ts, interval, pred_len, tz) -> pd.DatetimeIndex:
    """Generate intraday future bar timestamps, respecting session boundaries."""
    step_min = _step(interval)
    step_td = pd.Timedelta(minutes=step_min)

    # Ensure last_ts is tz-aware in the target tz
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize(tz)
    else:
        last_ts = last_ts.tz_convert(tz)

    results: list[pd.Timestamp] = []
    last_ts_utc = last_ts.tz_convert("UTC")

    # Find the session containing last_ts
    try:
        current_session = cal.minute_to_session(last_ts_utc, direction="previous")
    except Exception:
        current_session = cal.date_to_session(
            pd.Timestamp(last_ts.date()), direction="next"
        )

    next_bar = last_ts_utc + step_td
    max_sessions = pred_len + 30

    for _ in range(max_sessions):
        if len(results) >= pred_len:
            break

        session_open_utc = cal.session_open(current_session).tz_convert("UTC")
        session_close_utc = cal.session_close(current_session).tz_convert("UTC")

        # Snap to session open if the next bar falls before it
        if next_bar < session_open_utc:
            next_bar = session_open_utc

        # Skip this session entirely if next_bar is already past it
        if next_bar > session_close_utc:
            nxt = _next_session(cal, current_session)
            if nxt is None:
                break
            current_session = nxt
            next_bar = cal.session_open(current_session).tz_convert("UTC")
            continue

        # Walk through this session's bars
        while next_bar <= session_close_utc and len(results) < pred_len:
            results.append(next_bar.tz_convert(tz))
            next_bar = next_bar + step_td

        # Advance to the next session
        nxt = _next_session(cal, current_session)
        if nxt is None:
            break
        current_session = nxt
        next_bar = cal.session_open(current_session).tz_convert("UTC")

    return pd.DatetimeIndex(results)
