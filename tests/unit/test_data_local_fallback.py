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
