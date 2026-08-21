"""
Tests for the calendar-day window math in fetch_data_raw (strategy/kairos_strategies.py).

These are pure date/day-count math tests - no network calls, no GPU, no model loading.
They cover the bug where equities/ETFs/FX (which trade ~5/7 days a week) got a
calendar-day window sized as if they traded 24/7 like crypto, undershooting the real
bar count (e.g. "need 300 bars, got 287").
"""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pandas as pd

from kairos_strategies import (
    is_24_7_crypto_symbol, is_limited_hours_equity_symbol, calendar_days_for_bars,
    fetch_data_raw, KairosSettings, EQUITY_TRADING_HOURS_PER_DAY,
)


def test_crypto_symbols_are_24_7():
    assert is_24_7_crypto_symbol("BTC-USD")
    assert is_24_7_crypto_symbol("ETH-USD")


def test_equity_etf_symbols_are_not_24_7():
    assert not is_24_7_crypto_symbol("SPY")
    assert not is_24_7_crypto_symbol("QQQ")
    assert not is_24_7_crypto_symbol("DIA")
    assert not is_24_7_crypto_symbol("XLK")


def test_fx_symbols_are_not_24_7():
    assert not is_24_7_crypto_symbol("EURUSD=X")
    assert not is_24_7_crypto_symbol("GBPJPY=X")


def test_futures_symbols_are_not_24_7():
    assert not is_24_7_crypto_symbol("ES=F")
    assert not is_24_7_crypto_symbol("CL=F")


def test_crypto_gets_no_weekend_padding():
    # 1d interval => bars_per_day = 1. 300 bars needed.
    days = calendar_days_for_bars(bars_needed=300, bars_per_day=1, symbol="BTC-USD", buffer_days=30)
    assert days == 300 + 30


def test_equity_gets_7_over_5_padding():
    days_crypto = calendar_days_for_bars(bars_needed=300, bars_per_day=1, symbol="BTC-USD", buffer_days=30)
    days_equity = calendar_days_for_bars(bars_needed=300, bars_per_day=1, symbol="SPY", buffer_days=30)
    assert days_equity > days_crypto
    # raw padded days should be close to 300 * 7/5 + 5 = 425, plus buffer 30
    assert days_equity == int(300 * (7 / 5) + 5) + 30


def test_fx_gets_7_over_5_padding():
    days = calendar_days_for_bars(bars_needed=300, bars_per_day=1, symbol="EURUSD=X", buffer_days=30)
    assert days == int(300 * (7 / 5) + 5) + 30


def test_futures_gets_7_over_5_padding():
    days = calendar_days_for_bars(bars_needed=300, bars_per_day=1, symbol="ES=F", buffer_days=30)
    assert days == int(300 * (7 / 5) + 5) + 30


def test_padding_covers_previously_failing_case():
    # Previously observed failure: equities needed 300 bars but calendar window
    # (without padding) only yielded 287 real trading bars. The padded window
    # should request enough calendar days that at ~5/7 trading days we still
    # clear 300 real bars comfortably.
    days_equity = calendar_days_for_bars(bars_needed=300, bars_per_day=1, symbol="SPY", buffer_days=30)
    approx_trading_days = days_equity * (5 / 7)
    assert approx_trading_days >= 300


def test_limited_hours_equity_symbol_classification():
    assert is_limited_hours_equity_symbol("SPY")
    assert is_limited_hours_equity_symbol("AAPL")
    assert not is_limited_hours_equity_symbol("BTC-USD")
    assert not is_limited_hours_equity_symbol("EURUSD=X")
    assert not is_limited_hours_equity_symbol("ES=F")


def test_equity_intraday_gets_hours_per_day_correction():
    """Residual bug found 2026-08-21, live: the 7/5 weekend-only correction
    (from the earlier "need 300 bars, got 287" fix) is not enough at intraday
    granularity for equities specifically -- bars_per_day=24 (from
    BARS_PER_DAY["1h"]) assumes 24 HOURS of trading per day, which is only
    true for crypto (and a fair approximation for near-continuous FX/futures).
    An NYSE equity day only covers ~6.5 hours, not 24. Confirmed live:
    fetch_data_raw raised "Not enough data for CB: need 300 bars, got 252"
    under the weekend-only correction alone. Equity's requested calendar-day
    window must now be meaningfully larger than the naive (pre-fix) window at
    bars_per_day=24, to actually cover 300 real bars once trading-hours
    sparsity is accounted for."""
    naive_days = int(300 / 24 * (7 / 5) + 5) + 30  # the old (incomplete) formula
    fixed_days = calendar_days_for_bars(bars_needed=300, bars_per_day=24, symbol="CB", buffer_days=30)
    assert fixed_days > naive_days

    # Real bars covered by the fixed window: ~5/7 trading days/week x
    # ~6.5 bars/day (not 24) must clear the 300 bars requested.
    approx_trading_days = (fixed_days - 30) * (5 / 7)
    approx_real_bars = approx_trading_days * EQUITY_TRADING_HOURS_PER_DAY
    assert approx_real_bars >= 300


def test_fx_futures_intraday_unaffected_by_equity_hours_correction():
    """FX/futures trade near-continuously through the weekday session, so they
    should NOT get the equity-specific hours-per-day rescaling -- only the
    existing weekend-only correction, same as before this fix."""
    days_fx = calendar_days_for_bars(bars_needed=300, bars_per_day=24, symbol="EURUSD=X", buffer_days=30)
    days_futures = calendar_days_for_bars(bars_needed=300, bars_per_day=24, symbol="ES=F", buffer_days=30)
    expected = int(300 / 24 * (7 / 5) + 5) + 30
    assert days_fx == expected
    assert days_futures == expected


def test_crypto_intraday_unaffected_by_equity_hours_correction():
    """Crypto is 24/7 -- must stay completely unpadded, exactly as before."""
    days = calendar_days_for_bars(bars_needed=300, bars_per_day=24, symbol="BTC-USD", buffer_days=30)
    assert days == int(300 / 24) + 30


def test_equity_daily_interval_unaffected_by_hours_correction():
    """At bars_per_day=1 (daily interval), the hours-per-day correction must
    not kick in -- 1 bar/trading-day is already correct for equities at daily
    granularity, matching all the bars_per_day=1 tests above (unchanged)."""
    days_before_this_fix = int(300 * (7 / 5) + 5) + 30
    days_now = calendar_days_for_bars(bars_needed=300, bars_per_day=1, symbol="SPY", buffer_days=30)
    assert days_now == days_before_this_fix


def test_as_of_caps_fetch_window_end_date(monkeypatch):
    """as_of should replace date.today() as the fetch window's end date."""
    import kairos_strategies

    captured = {}

    def fake_get_price_data(symbol, start_date, end_date, interval):
        captured["end_date"] = end_date
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        return pd.DataFrame({"Close": range(10)}, index=idx)

    monkeypatch.setattr(kairos_strategies.price_cache, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(KairosSettings, "interval", "1d")

    as_of = datetime(2026, 1, 5, 12, 0)
    fetch_data_raw("BTC-USD", lookback=3, as_of=as_of)

    assert captured["end_date"] == "2026-01-05"


def test_as_of_drops_bars_after_cutoff(monkeypatch):
    """Bars timestamped after as_of must be dropped (round down to nearest bar)."""
    import kairos_strategies

    def fake_get_price_data(symbol, start_date, end_date, interval):
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        return pd.DataFrame({"Close": range(10)}, index=idx)

    monkeypatch.setattr(kairos_strategies.price_cache, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(KairosSettings, "interval", "1d")

    as_of = datetime(2026, 1, 5, 12, 0)
    raw = fetch_data_raw("BTC-USD", lookback=3, as_of=as_of)

    assert raw.index.max() <= as_of
    assert raw.index.max() == datetime(2026, 1, 5)


def test_no_as_of_preserves_existing_behavior(monkeypatch):
    """Without as_of, no post-fetch filtering is applied (existing behavior)."""
    import kairos_strategies

    def fake_get_price_data(symbol, start_date, end_date, interval):
        idx = pd.date_range("2026-01-01", periods=10, freq="D")
        return pd.DataFrame({"Close": range(10)}, index=idx)

    monkeypatch.setattr(kairos_strategies.price_cache, "get_price_data", fake_get_price_data)
    monkeypatch.setattr(KairosSettings, "interval", "1d")

    raw = fetch_data_raw("BTC-USD", lookback=3)

    assert len(raw) == 10


def _seed_local_prices_db(db_path, ticker="CRV-USD", start="2025-06-01", n_bars=20, interval_minutes=1440):
    """Create a throwaway local price_cache-shaped SQLite DB with real rows for
    `ticker`, mimicking the `prices` table fetch_price_data_local_fallback reads."""
    import sqlite3
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


class TestFetchDataRawLocalSqliteFallback:
    """price_cache can mark a ticker no-data after one failed remote fetch even
    though its local prices table already has good history -- observed in
    production: a finetune_next backtest crashed with "No price data returned
    for CRV-USD" moments after that exact symbol's *training* fetch succeeded
    via this same local-SQLite fallback earlier in the same run. fetch_data_raw
    must fall back to reading the local prices table directly before giving up,
    same as kairos/cli/finetune.py's training-data fetch already does."""

    def test_falls_back_to_local_sqlite_when_price_cache_returns_none(self, tmp_path, monkeypatch, capsys):
        import kairos_strategies

        as_of = datetime(2026, 1, 5, 12, 0)
        db_path = str(tmp_path / "prices.db")
        # fetch_data_raw's fallback window ends at as_of and reaches back
        # ~33 calendar days (lookback=3 + 30-day buffer, no weekend padding
        # for a 24/7 crypto symbol) -- seed inside that window.
        _seed_local_prices_db(db_path, ticker="CRV-USD", start="2025-12-10", n_bars=20)

        monkeypatch.setattr(kairos_strategies.price_cache, "get_price_data", lambda *a, **k: None)
        monkeypatch.setattr(kairos_strategies.price_cache, "DB_PATH", db_path)
        monkeypatch.setattr(KairosSettings, "interval", "1d")

        raw = fetch_data_raw("CRV-USD", lookback=3, as_of=as_of)

        assert len(raw) == 20
        assert "close" in raw.columns
        assert "using direct local SQLite fallback" in capsys.readouterr().out

    def test_fallback_warning_prints_only_once_per_symbol(self, tmp_path, monkeypatch, capsys):
        """fetch_data_raw is called once per (symbol, date) pair by callers like
        prewarm_prediction_cache, which sweeps every date in a backtest window.
        The fallback line must only print the first time a given symbol hits
        this path in the process, not on every call."""
        import kairos_strategies

        as_of = datetime(2026, 1, 5, 12, 0)
        db_path = str(tmp_path / "prices.db")
        _seed_local_prices_db(db_path, ticker="CRV-USD", start="2025-12-10", n_bars=20)

        monkeypatch.setattr(kairos_strategies.price_cache, "get_price_data", lambda *a, **k: None)
        monkeypatch.setattr(kairos_strategies.price_cache, "DB_PATH", db_path)
        monkeypatch.setattr(KairosSettings, "interval", "1d")
        kairos_strategies._no_data_fallback_warned.clear()

        raw1 = fetch_data_raw("CRV-USD", lookback=3, as_of=as_of)
        assert "using direct local SQLite fallback" in capsys.readouterr().out

        raw2 = fetch_data_raw("CRV-USD", lookback=3, as_of=as_of)
        assert "using direct local SQLite fallback" not in capsys.readouterr().out

        assert len(raw1) == 20
        assert len(raw2) == 20

        kairos_strategies._no_data_fallback_warned.clear()

    def test_raises_when_local_sqlite_also_has_no_data(self, tmp_path, monkeypatch):
        import kairos_strategies

        as_of = datetime(2026, 1, 5, 12, 0)
        db_path = str(tmp_path / "empty_prices.db")
        _seed_local_prices_db(db_path, ticker="SOME-OTHER-USD", start="2025-12-10", n_bars=5)

        monkeypatch.setattr(kairos_strategies.price_cache, "get_price_data", lambda *a, **k: None)
        monkeypatch.setattr(kairos_strategies.price_cache, "DB_PATH", db_path)
        monkeypatch.setattr(KairosSettings, "interval", "1d")

        try:
            fetch_data_raw("CRV-USD", lookback=3, as_of=as_of)
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "No price data returned for CRV-USD" in str(exc)
