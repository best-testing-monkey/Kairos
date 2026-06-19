"""Unit tests for kairos.windowing (KAI-3)."""
import pytest
import pandas as pd
import numpy as np

from kairos.windowing import estimate_start, fetch_with_retry
from kairos.errors import InsufficientDataError, UnsupportedIntervalError

TZ = "America/New_York"


def _ts(date: str) -> pd.Timestamp:
    return pd.Timestamp(date, tz=TZ)


class TestEstimateStart:
    def test_daily_returns_earlier_date(self):
        end = _ts("2024-01-31")
        start = estimate_start(end, "1d", 20)
        assert start < end

    def test_daily_lookback_span_covers_n_trading_days(self):
        end = _ts("2024-12-31")
        start = estimate_start(end, "1d", 252, buffer=2.0)
        # span should cover at least 252 calendar days worth of trading days
        span_days = (end - start).days
        assert span_days > 252

    def test_intraday_span_much_larger(self):
        end = _ts("2024-06-14")
        start_daily = estimate_start(end, "1d", 20)
        start_1m = estimate_start(end, "1m", 20)
        # 1m bars are densely packed in a day; start should be much closer
        assert (end - start_1m).days < (end - start_daily).days

    def test_unsupported_interval_raises(self):
        with pytest.raises(UnsupportedIntervalError):
            estimate_start(_ts("2024-01-01"), "99x", 10)


class TestFetchWithRetry:
    def _make_frame(self, n: int) -> pd.DataFrame:
        idx = pd.date_range("2023-01-02", periods=n, freq="B", tz=TZ)
        return pd.DataFrame(
            {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5,
             "Volume": 1000, "Dividends": 0, "Stock Splits": 0, "market_cap": 0},
            index=idx,
        )

    def test_succeeds_first_try(self):
        frame = self._make_frame(30)

        def fetch(sym, start, end, interval):
            return frame

        result = fetch_with_retry("AAPL", "1d", 20, _ts("2023-02-10"), fetch)
        assert len(result) >= 20

    def test_widens_on_short_frame(self):
        calls = []
        short = self._make_frame(5)
        full = self._make_frame(30)

        def fetch(sym, start, end, interval):
            calls.append((start, end))
            return short if len(calls) < 3 else full

        result = fetch_with_retry("AAPL", "1d", 20, _ts("2023-02-10"), fetch)
        assert len(calls) >= 2
        assert len(result) >= 20

    def test_raises_after_retry_cap(self):
        def fetch(sym, start, end, interval):
            return None

        with pytest.raises(InsufficientDataError) as exc_info:
            fetch_with_retry("AAPL", "1d", 20, _ts("2023-02-10"), fetch, retry_cap=2)
        assert exc_info.value.want == 20

    def test_at_most_retry_cap_plus_one_calls(self):
        calls = []

        def fetch(sym, start, end, interval):
            calls.append(1)
            return None

        with pytest.raises(InsufficientDataError):
            fetch_with_retry("AAPL", "1d", 20, _ts("2023-02-10"), fetch, retry_cap=3)

        assert len(calls) == 4  # retry_cap + 1
