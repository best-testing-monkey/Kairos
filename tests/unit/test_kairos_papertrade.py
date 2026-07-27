import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import json
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from kairos_papertrade import (
    parse_report_effective_dt,
    generate_and_dedupe_reports,
    map_instrument_type,
    compute_pct_profit_per_trade,
    write_json_report,
    _build_arg_parser,
    _IntradayFallbackProvider,
    _INTRADAY_FALLBACK_LADDER,
    _notify,
    _format_start_message,
    _format_finish_message,
    _format_crash_message,
)
import kairos_papertrade as kp
from kairos.ops import OpsError


# ============================================================================
# parse_report_effective_dt
# ============================================================================

class TestParseReportEffectiveDt:
    def test_parses_valid_header(self, tmp_path):
        report = tmp_path / "kairos_signals_202607192052.md"
        report.write_text(
            "# Kairos Signals Report 2026-07-19 2052h\n\n_Filters: min ev_pct 0.10%_\n"
        )
        result = parse_report_effective_dt(str(report))
        assert result == datetime(2026, 7, 19, 20, 52)

    def test_parses_midnight_header(self, tmp_path):
        report = tmp_path / "kairos_signals_202607110000.md"
        report.write_text("# Kairos Signals Report 2026-07-11 0000h\n")
        result = parse_report_effective_dt(str(report))
        assert result == datetime(2026, 7, 11, 0, 0)

    def test_never_falls_back_to_filename(self, tmp_path):
        # Filename says one date, but header (the source of truth) is
        # missing/garbled -- must raise, not silently trust the filename.
        report = tmp_path / "kairos_signals_202601010000.md"
        report.write_text("Some unrelated first line\n\n# Kairos Signals Report 2026-07-19 2052h\n")
        with pytest.raises(ValueError):
            parse_report_effective_dt(str(report))

    def test_rejects_malformed_header(self, tmp_path):
        report = tmp_path / "bad.md"
        report.write_text("# Kairos Signals Report not-a-date\n")
        with pytest.raises(ValueError):
            parse_report_effective_dt(str(report))

    def test_rejects_empty_file(self, tmp_path):
        report = tmp_path / "empty.md"
        report.write_text("")
        with pytest.raises(ValueError):
            parse_report_effective_dt(str(report))


# ============================================================================
# generate_and_dedupe_reports
# ============================================================================

class TestGenerateAndDedupeReports:
    def test_dedupes_by_effective_dt_first_seen_wins(self, tmp_path, monkeypatch):
        # Simulate a weekend: several distinct `now` values (iter_now) all
        # resolve to the SAME last-closed-bar effective_dt in their report
        # header (e.g. Sat/Sun/Mon-morning all show Friday's close).
        calls = []

        def fake_run(now, intervals, return_rows, **kwargs):
            calls.append(now)
            # First two calls collapse to the same effective_dt (weekend
            # dup); the third is a distinct, older effective_dt.
            if len(calls) <= 2:
                effective = "2026-07-17 0000h"
            else:
                effective = "2026-07-16 0000h"
            report_path = tmp_path / f"report_{len(calls)}.md"
            report_path.write_text(f"# Kairos Signals Report {effective}\n")
            stats_rows = [{"call": len(calls)}]
            advice_rows = [{"call": len(calls)}]
            return str(report_path), stats_rows, advice_rows

        import kairos_papertrade as kp
        monkeypatch.setattr(kp._kairos_signals_mod, "run", fake_run)

        base_now = datetime(2026, 7, 19, 0, 0)
        result = generate_and_dedupe_reports(base_now, "1d", months_back=0.1, run_kwargs={})

        # 0.1 months * 30.44 ~= 3.044 -> round() -> 3 iterations
        assert len(calls) == 3
        # Only 2 distinct effective_dts should survive de-dup.
        assert len(result) == 2
        # Sorted oldest-first.
        assert result[0][0] < result[1][0]
        assert result[0][0] == datetime(2026, 7, 16, 0, 0)
        assert result[1][0] == datetime(2026, 7, 17, 0, 0)
        # First-seen wins: effective_dt 2026-07-17 should keep call #1's rows.
        assert result[1][1] == [{"call": 1}]

    def test_returns_empty_list_when_no_iterations(self, monkeypatch):
        import kairos_papertrade as kp

        def fake_run(now, intervals, return_rows, **kwargs):
            raise AssertionError("run() should never be called for 0 iterations")

        monkeypatch.setattr(kp._kairos_signals_mod, "run", fake_run)
        base_now = datetime(2026, 7, 19, 0, 0)
        result = generate_and_dedupe_reports(base_now, "1d", months_back=0.0, run_kwargs={})
        assert result == []

    def _fake_run_factory(self, tmp_path):
        calls = []

        def fake_run(now, intervals, return_rows, **kwargs):
            calls.append(now)
            report_path = tmp_path / f"report_{len(calls)}.md"
            report_path.write_text(
                f"# Kairos Signals Report {now:%Y-%m-%d} 0000h\n"
            )
            return str(report_path), [{"call": len(calls)}], [{"call": len(calls)}]

        return calls, fake_run

    def test_slow_iteration_sends_watchdog_notification(self, tmp_path, monkeypatch):
        import kairos_papertrade as kp

        calls, fake_run = self._fake_run_factory(tmp_path)
        monkeypatch.setattr(kp._kairos_signals_mod, "run", fake_run)
        mock_send = MagicMock()
        monkeypatch.setattr(kp, "send_telegram", mock_send)

        # Each iteration's elapsed = monotonic() (after call) - monotonic()
        # (before call). Two monotonic() reads happen per loop iteration;
        # feed pairs that are exactly 301s apart so every iteration is "slow".
        times = iter(t for pair in ((0.0, 301.0), (301.0, 602.0), (602.0, 903.0))
                      for t in pair)
        monkeypatch.setattr(kp.time, "monotonic", lambda: next(times))

        base_now = datetime(2026, 7, 19, 0, 0)
        result = generate_and_dedupe_reports(
            base_now, "1d", months_back=0.1, run_kwargs={}, notify=True,
        )

        assert len(calls) == 3
        assert mock_send.call_count == 3
        for c in mock_send.call_args_list:
            msg = c.args[0]
            assert msg.startswith("⏱️")
            assert "5.0min" in msg
            assert c.kwargs.get("parse_mode") is None

    def test_fast_iteration_sends_no_watchdog_notification(self, tmp_path, monkeypatch):
        import kairos_papertrade as kp

        calls, fake_run = self._fake_run_factory(tmp_path)
        monkeypatch.setattr(kp._kairos_signals_mod, "run", fake_run)
        mock_send = MagicMock()
        monkeypatch.setattr(kp, "send_telegram", mock_send)

        # 1 second per iteration -- well under the 300s threshold.
        times = iter(t for pair in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0)) for t in pair)
        monkeypatch.setattr(kp.time, "monotonic", lambda: next(times))

        base_now = datetime(2026, 7, 19, 0, 0)
        result = generate_and_dedupe_reports(
            base_now, "1d", months_back=0.1, run_kwargs={}, notify=True,
        )

        assert len(calls) == 3
        mock_send.assert_not_called()

    def test_slow_iteration_notification_suppressed_when_notify_false(self, tmp_path, monkeypatch):
        import kairos_papertrade as kp

        calls, fake_run = self._fake_run_factory(tmp_path)
        monkeypatch.setattr(kp._kairos_signals_mod, "run", fake_run)
        mock_send = MagicMock()
        monkeypatch.setattr(kp, "send_telegram", mock_send)

        times = iter(t for pair in ((0.0, 301.0), (301.0, 602.0), (602.0, 903.0))
                      for t in pair)
        monkeypatch.setattr(kp.time, "monotonic", lambda: next(times))

        base_now = datetime(2026, 7, 19, 0, 0)
        generate_and_dedupe_reports(
            base_now, "1d", months_back=0.1, run_kwargs={}, notify=False,
        )

        mock_send.assert_not_called()


# ============================================================================
# map_instrument_type
# ============================================================================

class TestMapInstrumentType:
    def test_futures_ticker_is_cfd(self):
        assert map_instrument_type({"ticker": "CL=F", "direction": "long"}) == "cfd"
        assert map_instrument_type({"ticker": "NG=F", "direction": "long"}) == "cfd"

    def test_crypto_ticker_is_cfd(self):
        assert map_instrument_type({"ticker": "WIF-USD", "direction": "long"}) == "cfd"
        assert map_instrument_type({"ticker": "UNI7083-USD", "direction": "long"}) == "cfd"

    def test_forex_ticker_is_cfd(self):
        assert map_instrument_type({"ticker": "EURUSD=X", "direction": "long"}) == "cfd"

    def test_plain_equity_ticker_is_stock(self):
        assert map_instrument_type({"ticker": "AAPL", "direction": "long"}) == "stock"
        assert map_instrument_type({"ticker": "NFLX", "direction": "long"}) == "stock"

    def test_short_direction_is_always_cfd(self):
        assert map_instrument_type({"ticker": "AAPL", "direction": "short"}) == "cfd"

    def test_accepts_attribute_object_not_just_dict(self):
        from types import SimpleNamespace
        cand = SimpleNamespace(ticker="AAPL", direction="long")
        assert map_instrument_type(cand) == "stock"
        cand_short = SimpleNamespace(ticker="TSLA", direction="short")
        assert map_instrument_type(cand_short) == "cfd"


# ============================================================================
# compute_pct_profit_per_trade
# ============================================================================

class TestComputePctProfitPerTrade:
    def test_computes_mean_pct_across_positions(self):
        # Two fake closed positions: +10% and -5% (of entry_price*quantity).
        positions = [
            {"realized_pnl": 100.0, "entry_price": 10.0, "quantity": 100.0},  # 100/1000 = 10%
            {"realized_pnl": -25.0, "entry_price": 5.0, "quantity": 100.0},   # -25/500 = -5%
        ]
        result = compute_pct_profit_per_trade(positions)
        assert result == pytest.approx((10.0 + -5.0) / 2)

    def test_ignores_positions_missing_fields(self):
        positions = [
            {"realized_pnl": None, "entry_price": 10.0, "quantity": 100.0},
            {"realized_pnl": 100.0, "entry_price": 10.0, "quantity": 100.0},
        ]
        result = compute_pct_profit_per_trade(positions)
        assert result == pytest.approx(10.0)

    def test_returns_none_for_no_positions(self):
        assert compute_pct_profit_per_trade([]) is None

    def test_returns_none_when_all_positions_uncomputable(self):
        positions = [{"realized_pnl": None, "entry_price": 0.0, "quantity": 0.0}]
        assert compute_pct_profit_per_trade(positions) is None

    def test_accepts_attribute_objects(self):
        from types import SimpleNamespace
        positions = [
            SimpleNamespace(realized_pnl=50.0, entry_price=10.0, quantity=50.0),  # 50/500=10%
        ]
        assert compute_pct_profit_per_trade(positions) == pytest.approx(10.0)


# ============================================================================
# write_json_report
# ============================================================================

class TestWriteJsonReport:
    def test_writes_expected_shape(self, tmp_path):
        metrics = {
            "total_profit_eur": 12.5,
            "pct_profit": 6.25,
            "pct_profit_per_trade": 1.5,
            "pct_max_drawdown": 3.2,
            "sharpe": 0.8,
            "num_trades": 4,
        }
        meta = {"start": "2026-07-01T00:00:00", "end": "2026-07-19T00:00:00", "interval": "1d"}
        out_path = tmp_path / "report.json"

        result_path = write_json_report(metrics, meta, out_path)

        assert os.path.exists(result_path)
        with open(result_path) as f:
            payload = json.load(f)

        for key in ("total_profit_eur", "pct_profit", "pct_profit_per_trade",
                    "pct_max_drawdown", "sharpe", "num_trades"):
            assert key in payload
            assert payload[key] == metrics[key]

        assert "meta" in payload
        assert payload["meta"] == meta

    def test_creates_parent_directories(self, tmp_path):
        out_path = tmp_path / "nested" / "dir" / "report.json"
        write_json_report({"num_trades": 0}, {}, out_path)
        assert out_path.exists()


# ============================================================================
# _build_arg_parser
# ============================================================================

class TestBuildArgParser:
    def test_min_ev_pct_default_is_015(self):
        # Verify that the --min_ev_pct default matches the measured realized
        # cost, per docs/papertrade_loss_analysis.md Factor 7.
        args = _build_arg_parser().parse_args([])
        assert args.min_ev_pct == 0.15


# ============================================================================
# _IntradayFallbackProvider
# ============================================================================

def _make_bars_df(hour_utc_naive_ny=10):
    # A single bar timestamped mid-day in naive America/New_York local time,
    # matching what price_cache.get_price_data returns before the
    # tz-localize/convert post-processing in get_bars().
    idx = pd.DatetimeIndex([datetime(2026, 7, 20, hour_utc_naive_ny, 0)])
    return pd.DataFrame(
        {
            "Open": [100.0], "High": [101.0], "Low": [99.0],
            "Close": [100.5], "Volume": [1000],
        },
        index=idx,
    )


class TestIntradayFallbackProvider:
    def test_first_ladder_interval_returned_directly_single_call(self, tmp_path, monkeypatch):
        calls = []

        def fake_get_price_data(ticker, start_date, end_date, interval="1d", db_path=None):
            calls.append(interval)
            return _make_bars_df()

        monkeypatch.setattr(kp.price_cache, "get_price_data", fake_get_price_data)
        monkeypatch.setattr(kp.price_cache, "configure", lambda **kw: None)

        provider = _IntradayFallbackProvider(str(tmp_path))
        start = datetime(2026, 7, 20)
        end = datetime(2026, 7, 21)
        result = provider.get_bars("AAPL", start, end)

        assert calls == ["1m"]
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert str(result.index.tz) == "UTC"
        assert len(result) == 1
        assert result["Close"].iloc[0] == pytest.approx(100.5)

    def test_falls_through_ladder_to_first_interval_with_data(self, tmp_path, monkeypatch):
        calls = []

        def fake_get_price_data(ticker, start_date, end_date, interval="1d", db_path=None):
            calls.append(interval)
            if interval == "1h":
                return _make_bars_df()
            return None

        monkeypatch.setattr(kp.price_cache, "get_price_data", fake_get_price_data)
        monkeypatch.setattr(kp.price_cache, "configure", lambda **kw: None)

        provider = _IntradayFallbackProvider(str(tmp_path))
        start = datetime(2026, 7, 20)
        end = datetime(2026, 7, 21)
        result = provider.get_bars("AAPL", start, end)

        assert calls == _INTRADAY_FALLBACK_LADDER  # walked the full ladder in order
        assert len(result) == 1
        assert result["Close"].iloc[0] == pytest.approx(100.5)

    def test_all_intraday_empty_delegates_to_fallback_1d(self, tmp_path, monkeypatch):
        def fake_get_price_data(ticker, start_date, end_date, interval="1d", db_path=None):
            return None  # every intraday interval comes back empty

        monkeypatch.setattr(kp.price_cache, "get_price_data", fake_get_price_data)
        monkeypatch.setattr(kp.price_cache, "configure", lambda **kw: None)

        provider = _IntradayFallbackProvider(str(tmp_path))
        sentinel = pd.DataFrame({"Open": [1.0]})
        provider._fallback.get_bars = MagicMock(return_value=sentinel)

        start = datetime(2026, 7, 20)
        end = datetime(2026, 7, 21)
        result = provider.get_bars("AAPL", start, end)

        provider._fallback.get_bars.assert_called_once_with("AAPL", start, end)
        assert result is sentinel

    def test_get_current_price_delegates_to_fallback(self, tmp_path):
        provider = _IntradayFallbackProvider(str(tmp_path))
        provider._fallback.get_current_price = MagicMock(return_value=42.0)
        assert provider.get_current_price("AAPL") == 42.0
        provider._fallback.get_current_price.assert_called_once_with("AAPL")

    def test_get_bid_ask_delegates_to_fallback(self, tmp_path):
        provider = _IntradayFallbackProvider(str(tmp_path))
        provider._fallback.get_bid_ask = MagicMock(return_value=(99.5, 100.5))
        assert provider.get_bid_ask("AAPL") == (99.5, 100.5)
        provider._fallback.get_bid_ask.assert_called_once_with("AAPL")

    def test_get_dividends_delegates_to_fallback(self, tmp_path):
        provider = _IntradayFallbackProvider(str(tmp_path))
        start = datetime(2026, 7, 1)
        end = datetime(2026, 7, 31)
        provider._fallback.get_dividends = MagicMock(return_value=[])
        result = provider.get_dividends("AAPL", start, end)
        assert result == []
        provider._fallback.get_dividends.assert_called_once_with("AAPL", start, end)


# ============================================================================
# _notify (Telegram helper) and --no-telegram flag
# ============================================================================

class TestNotify:
    def test_enabled_calls_send_telegram_with_plain_text(self, monkeypatch):
        mock_send = MagicMock()
        monkeypatch.setattr(kp, "send_telegram", mock_send)
        _notify("hello", enabled=True)
        mock_send.assert_called_once_with("hello", parse_mode=None)

    def test_disabled_is_a_silent_noop(self, monkeypatch):
        mock_send = MagicMock()
        monkeypatch.setattr(kp, "send_telegram", mock_send)
        _notify("hello", enabled=False)
        mock_send.assert_not_called()

    def test_ops_error_is_swallowed_not_raised(self, monkeypatch):
        mock_send = MagicMock(side_effect=OpsError("no creds"))
        monkeypatch.setattr(kp, "send_telegram", mock_send)
        # Must not raise.
        _notify("hello", enabled=True)
        mock_send.assert_called_once()

    def test_default_enabled_is_true(self, monkeypatch):
        mock_send = MagicMock()
        monkeypatch.setattr(kp, "send_telegram", mock_send)
        _notify("hello")
        mock_send.assert_called_once_with("hello", parse_mode=None)


class TestNoTelegramFlag:
    def test_notify_defaults_true(self):
        args = _build_arg_parser().parse_args([])
        assert args.notify is True

    def test_no_telegram_sets_notify_false(self):
        args = _build_arg_parser().parse_args(["--no-telegram"])
        assert args.notify is False


# ============================================================================
# Message formatting helpers
# ============================================================================

class TestFormatStartMessage:
    def test_contains_emoji_and_key_params(self):
        args = _build_arg_parser().parse_args(
            ["--interval", "1d", "--months-back", "3", "--top-n", "5",
             "--capital", "500", "--broker", "IBKR"]
        )
        base_now = datetime(2026, 7, 27, 12, 30)
        msg = _format_start_message(base_now, args)
        assert msg.startswith("🟢")
        assert "starting" in msg
        assert "2026-07-27 12:30" in msg
        assert "interval=1d" in msg
        assert "months_back=3.0" in msg
        assert "top_n=5" in msg
        assert "capital=500.0" in msg
        assert "broker=IBKR" in msg


class TestFormatFinishMessage:
    def test_contains_emoji_and_rounded_metrics(self):
        metrics = {
            "total_profit_eur": 12.3456,
            "pct_profit": 6.789,
            "pct_profit_per_trade": 1.5,
            "pct_max_drawdown": 3.2,
            "sharpe": 0.812345,
            "num_trades": 7,
        }
        msg = _format_finish_message(metrics, "kairos_signals_papertrade_x.json")
        assert msg.startswith("✅")
        assert "finished" in msg
        assert "12.35" in msg
        assert "6.79" in msg
        assert "0.81" in msg
        assert "num_trades=7" in msg
        assert "kairos_signals_papertrade_x.json" in msg
        # Must not leak a full path -- caller passes os.path.basename already.
        assert "/" not in msg.split("Report: ")[1]


class TestFormatCrashMessage:
    def test_contains_emoji_exception_type_and_traceback_tail(self):
        args = _build_arg_parser().parse_args(["--interval", "1h"])
        base_now = datetime(2026, 7, 27, 8, 0)
        try:
            raise ValueError("boom")
        except ValueError as exc:
            msg = _format_crash_message(exc, base_now, args)
        assert msg.startswith("💥")
        assert "CRASHED" in msg
        assert "ValueError" in msg
        assert "interval=1h" in msg
        assert "boom" in msg
        assert "```" in msg
