"""Tests for kairos.data's local-SQLite fallback when price_cache.get_price_data
returns None.

price_cache marks a whole ticker as no-data in its no_data_tickers table after
a single failed fetch (e.g. a keyless provider), which can hide data that is
already sitting in the local prices table. This bit strategy/kairos_strategies.py's
fetch_data_raw first (a finetune_next backtest crashed on a symbol whose
training fetch had just succeeded via this exact fallback moments earlier in
the same run); kairos.data.get_forecast_window -- the live signal-generation
path -- had the identical gap and is covered here.
"""
import sqlite3

import pandas as pd
import pytest
from unittest.mock import patch

import kairos
from kairos.data import fetch_price_data_local_fallback


def _seed_local_prices_db(db_path, ticker, start, n_bars, interval_minutes=1440):
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE prices (
             ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
             volume REAL, dividends REAL, stock_splits REAL, market_cap REAL,
             interval_minutes INTEGER
        )"""
    )
    idx = pd.date_range(start, periods=n_bars, freq="D")
    for i, d in enumerate(idx):
        conn.execute(
            "INSERT INTO prices VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ticker, d.strftime("%Y-%m-%d"), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i,
             1000.0, 0.0, 0.0, None, interval_minutes),
        )
    conn.commit()
    conn.close()


def _seed_local_prices_db_hourly(db_path, ticker, start_utc, n_bars, interval_minutes=60):
    """Seed database with hourly data (interval_minutes=60).

    start_utc: Start time as UTC string (e.g., "2025-11-02 04:00:00").
    Creates naive datetime strings in the DB that, when tz_localize'd to Eastern,
    will include the ambiguous DST fall-back hour.
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE prices (
             ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
             volume REAL, dividends REAL, stock_splits REAL, market_cap REAL,
             interval_minutes INTEGER
        )"""
    )
    # Create UTC index
    idx = pd.date_range(start_utc, periods=n_bars, freq="h", tz="UTC")
    # Convert to naive Eastern time (as if read from DB storing Eastern times)
    idx_eastern_naive = idx.tz_convert("America/New_York").tz_localize(None)
    for i, d in enumerate(idx_eastern_naive):
        conn.execute(
            "INSERT INTO prices VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ticker, d.strftime("%Y-%m-%d %H:%M:%S"), 100.0 + i, 101.0 + i, 99.0 + i,
             100.5 + i, 1000.0, 0.0, 0.0, None, interval_minutes),
        )
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def configure_kairos():
    kairos.configure(remote=False)


class TestFetchPriceDataLocalFallback:
    """Direct tests of the shared helper itself."""

    def test_returns_rows_within_window(self, tmp_path):
        db_path = str(tmp_path / "prices.db")
        _seed_local_prices_db(db_path, "CRV-USD", "2025-12-10", n_bars=20)

        raw = fetch_price_data_local_fallback(
            "CRV-USD", pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-05"),
            "1d", db_path,
        )

        assert raw is not None
        assert len(raw) == 20
        assert list(raw.columns) == [
            "Open", "High", "Low", "Close", "Volume", "Dividends", "Stock Splits", "market_cap",
        ]

    def test_returns_none_for_unknown_ticker(self, tmp_path):
        db_path = str(tmp_path / "prices.db")
        _seed_local_prices_db(db_path, "CRV-USD", "2025-12-10", n_bars=20)

        raw = fetch_price_data_local_fallback(
            "SOME-OTHER-USD", pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-05"),
            "1d", db_path,
        )
        assert raw is None

    def test_returns_none_when_db_missing(self, tmp_path):
        raw = fetch_price_data_local_fallback(
            "CRV-USD", pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-05"),
            "1d", str(tmp_path / "does_not_exist.db"),
        )
        assert raw is None

    def test_handles_dst_ambiguous_time_hourly(self, tmp_path):
        """Test that hourly fetches across US DST fall-back date (2025-11-02)
        do not raise "Cannot infer dst time" error.

        On 2025-11-02 at 02:00 EDT, clocks fall back to 01:00 EST, making
        the wall-clock hour 01:00 occur twice (first in EDT, then in EST).
        When data stored as naive Eastern times includes this repeated hour,
        tz_localize with ambiguous="infer" must correctly disambiguate based
        on the monotonically increasing order of the UTC source data.
        """
        db_path = str(tmp_path / "prices_hourly.db")
        # Create hourly data from UTC times that span the DST fall-back transition.
        # Starting from 2025-11-02 04:00 UTC:
        #   04:00 UTC = 00:00 EDT
        #   05:00 UTC = 01:00 EDT
        #   06:00 UTC = 01:00 EST (repeated, but order makes it inferable)
        #   07:00 UTC = 02:00 EST
        _seed_local_prices_db_hourly(db_path, "BTC-USD", "2025-11-02 04:00:00", n_bars=7)

        raw = fetch_price_data_local_fallback(
            "BTC-USD", pd.Timestamp("2025-11-01"), pd.Timestamp("2025-11-03"),
            "1h", db_path,
        )

        assert raw is not None
        assert len(raw) == 7
        # Index must be tz-aware
        assert raw.index.tzinfo is not None
        # Index must be monotonically increasing (no DST ambiguity error)
        assert raw.index.is_monotonic_increasing


class TestGetForecastWindowLocalFallback:
    """get_forecast_window (the live signal-generation path) must fall back to
    the local prices table when price_cache.get_price_data returns None,
    exactly like fetch_data_raw and the finetune-training loader already do."""

    def test_falls_back_when_price_cache_returns_none(self, tmp_path, monkeypatch, capsys):
        db_path = str(tmp_path / "prices.db")
        _seed_local_prices_db(db_path, "CRV-USD", "2025-12-20", n_bars=30)
        monkeypatch.setattr("price_cache.DB_PATH", db_path)

        with patch("price_cache.get_price_data", return_value=None):
            x_df, x_ts, y_ts = kairos.get_forecast_window(
                "CRV-USD", "1d", lookback=5, pred_len=1, end="2026-01-10",
            )

        assert len(x_df) == 5
        assert "using direct local SQLite fallback" in capsys.readouterr().out

    def test_still_raises_no_data_error_when_local_fallback_also_empty(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "empty_prices.db")
        _seed_local_prices_db(db_path, "SOME-OTHER-USD", "2025-12-20", n_bars=30)
        monkeypatch.setattr("price_cache.DB_PATH", db_path)

        with patch("price_cache.get_price_data", return_value=None):
            with pytest.raises(kairos.NoDataError):
                kairos.get_forecast_window(
                    "CRV-USD", "1d", lookback=5, pred_len=1, end="2026-01-10",
                )
