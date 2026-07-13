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
    is_24_7_crypto_symbol, calendar_days_for_bars, fetch_data_raw, KairosSettings,
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
