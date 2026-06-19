"""Unit tests for kairos.calendars (KAI-4).

All fixtures are pinned to 2024 so holiday assertions remain stable.
"""
import pytest
import pandas as pd

from kairos.calendars import future_timestamps
from kairos.errors import CalendarError

TZ = "America/New_York"
CAL = "XNYS"  # NYSE


def _ts(dt: str, tz: str = TZ) -> pd.Timestamp:
    return pd.Timestamp(dt, tz=tz)


class TestDailyFuture:
    def test_friday_to_monday(self):
        """Daily: last bar on Friday → next bar is Monday (assuming no holiday)."""
        # 2024-01-05 is a Friday; 2024-01-08 is Monday (no holiday)
        last = _ts("2024-01-05")
        y = future_timestamps(last, "1d", 1, CAL, TZ)
        assert len(y) == 1
        assert y.iloc[0].date() == pd.Timestamp("2024-01-08").date()

    def test_skips_holiday(self):
        """Daily: MLK Day (2024-01-15, Monday) should be skipped."""
        # Last bar: 2024-01-12 (Friday); next session should be 2024-01-16 (Tuesday)
        last = _ts("2024-01-12")
        y = future_timestamps(last, "1d", 1, CAL, TZ)
        assert y.iloc[0].date() == pd.Timestamp("2024-01-16").date()

    def test_length_equals_pred_len(self):
        last = _ts("2024-01-05")
        for pred_len in (1, 5, 10):
            y = future_timestamps(last, "1d", pred_len, CAL, TZ)
            assert len(y) == pred_len

    def test_strictly_increasing(self):
        last = _ts("2024-01-05")
        y = future_timestamps(last, "1d", 10, CAL, TZ)
        assert (y.diff().iloc[1:] > pd.Timedelta(0)).all()

    def test_first_after_last(self):
        last = _ts("2024-01-05")
        y = future_timestamps(last, "1d", 5, CAL, TZ)
        assert y.iloc[0] > last

    def test_tz_matches(self):
        last = _ts("2024-01-05")
        y = future_timestamps(last, "1d", 5, CAL, TZ)
        assert str(y.iloc[0].tzinfo) == TZ

    def test_dst_spring_forward_2024(self):
        """DST spring forward: 2024-03-10 (Sunday). Timestamps around it must be valid."""
        last = _ts("2024-03-08")  # Friday before DST
        y = future_timestamps(last, "1d", 5, CAL, TZ)
        assert len(y) == 5
        assert (y.diff().iloc[1:] > pd.Timedelta(0)).all()


class TestIntradayFuture:
    def test_intraday_length(self):
        last = _ts("2024-01-05 15:55:00")
        y = future_timestamps(last, "5m", 10, CAL, TZ)
        assert len(y) == 10

    def test_last_bar_of_session_wraps_to_next_open(self):
        """5m: bar at 16:00 → next bar is at next-session open (09:30)."""
        last = _ts("2024-01-05 16:00:00")
        y = future_timestamps(last, "5m", 1, CAL, TZ)
        assert y.iloc[0].date() == pd.Timestamp("2024-01-08").date()
        assert y.iloc[0].hour == 9
        assert y.iloc[0].minute == 30

    def test_intraday_strictly_increasing(self):
        last = _ts("2024-01-05 14:00:00")
        y = future_timestamps(last, "5m", 20, CAL, TZ)
        assert (y.diff().iloc[1:] > pd.Timedelta(0)).all()

    def test_intraday_first_after_last(self):
        last = _ts("2024-01-05 14:00:00")
        y = future_timestamps(last, "5m", 5, CAL, TZ)
        assert y.iloc[0] > last


class TestErrors:
    def test_bad_calendar_raises(self):
        with pytest.raises(CalendarError):
            future_timestamps(_ts("2024-01-05"), "1d", 5, "XBAD_FAKE", TZ)
