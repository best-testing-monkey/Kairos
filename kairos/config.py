"""KAI-1: Configuration facade over price_cache.

One configure() call sets up both price_cache and the default exchange
calendar / timezone used by the rest of Kairos.
"""
from __future__ import annotations

import price_cache
from price_cache._config import load_config

from .errors import ConfigError

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------

_SUPPORTED_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h",
    "1d", "5d", "1wk", "1mo", "3mo",
}


class _State:
    calendar: str = "XNYS"
    tz: str = "America/New_York"


_state = _State()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure(
    *,
    remote: bool = False,
    local_mirror_path: str | None = None,
    calendar: str = "XNYS",
    tz: str = "America/New_York",
    config_file: str | None = None,
) -> None:
    """Configure Kairos and the underlying price_cache in one call.

    Args:
        remote: Passed through to price_cache.configure().
        local_mirror_path: Optional three-tier mirror path for price_cache.
        calendar: exchange_calendars calendar code (e.g. "XNYS", "XLON").
        tz: Default timezone for timestamps (IANA name).
        config_file: Optional path to a JSON/JSON5 config file for price_cache.
    """
    _validate_calendar(calendar)

    if config_file:
        cfg = load_config(config_file)
        price_cache.reconfigure_from_config(cfg)

    price_cache.configure(remote=remote, local_mirror_path=local_mirror_path)

    _state.calendar = calendar
    _state.tz = tz


def _validate_calendar(calendar: str) -> None:
    """Raise ConfigError if *calendar* is not known to exchange_calendars."""
    try:
        import exchange_calendars as xcals
        xcals.get_calendar(calendar)
    except Exception as exc:
        raise ConfigError(f"Unknown exchange calendar {calendar!r}: {exc}") from exc
