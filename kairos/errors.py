"""KAI error hierarchy - every failure mode before data reaches the model."""


class KairosError(Exception):
    """Base class for all Kairos errors."""


class ConfigError(KairosError):
    """Bad calendar, timezone, or config-file value."""


class UnsupportedIntervalError(KairosError):
    """Interval string not accepted by price_cache."""


class NoDataError(KairosError):
    """price_cache returned None for the requested symbol/range."""

    def __init__(self, symbol: str, start: str, end: str, interval: str):
        super().__init__(
            f"No data for {symbol!r} [{start} – {end}] interval={interval!r}"
        )
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval


class InsufficientDataError(KairosError):
    """Fewer bars available than the requested lookback, even after retries."""

    def __init__(self, have: int, want: int):
        super().__init__(f"Need {want} bars but only {have} available")
        self.have = have
        self.want = want


class DataQualityError(KairosError):
    """NaN, non-monotonic timestamps, or duplicate rows in the adapted frame."""


class CalendarError(KairosError):
    """Future timestamp generation failed (bad calendar, empty schedule, etc.)."""
