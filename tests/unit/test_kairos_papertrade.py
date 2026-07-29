import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import json
from datetime import datetime, timedelta
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
    _ensure_pred_cache_dir_env,
    _log_watchdog_snapshot,
    _read_self_rss_kb,
    DEFAULT_PRED_CACHE_DIR,
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

    def test_passes_on_group_timing_callback_per_iteration(self, tmp_path, monkeypatch):
        import kairos_papertrade as kp

        received_cbs = []

        def fake_run(now, intervals, return_rows, on_group_timing=None, **kwargs):
            received_cbs.append(on_group_timing)
            report_path = tmp_path / f"report_{len(received_cbs)}.md"
            report_path.write_text(f"# Kairos Signals Report {now:%Y-%m-%d} 0000h\n")
            return str(report_path), [], []

        monkeypatch.setattr(kp._kairos_signals_mod, "run", fake_run)

        base_now = datetime(2026, 7, 19, 0, 0)
        generate_and_dedupe_reports(base_now, "1d", months_back=0.1, run_kwargs={})

        assert len(received_cbs) == 3
        assert all(callable(cb) for cb in received_cbs)


# ============================================================================
# _log_group_timing / _make_group_timing_cb
# ============================================================================

class TestLogGroupTiming:
    def test_appends_expected_fields(self, monkeypatch, tmp_path):
        import kairos_papertrade as kp

        log_path = str(tmp_path / "papertrade_watchdog.log")
        monkeypatch.setattr(kp, "WATCHDOG_LOG_PATH", log_path)

        kp._log_group_timing(
            datetime(2026, 7, 19), "BTC-USD,ETH-USD", "1d", "Finetuned(BTC-USD,ETH-USD)",
            42.5, False,
        )

        with open(log_path) as f:
            content = f.read()
        assert "date=2026-07-19" in content
        assert f"pid={os.getpid()}" in content
        assert "model=Finetuned(BTC-USD,ETH-USD)" in content
        assert "interval=1d" in content
        assert "assets=BTC-USD,ETH-USD" in content
        assert "elapsed=42.5s" in content
        assert "cache=MISS" in content

    def test_cache_hit_logged_as_hit(self, monkeypatch, tmp_path):
        import kairos_papertrade as kp

        log_path = str(tmp_path / "papertrade_watchdog.log")
        monkeypatch.setattr(kp, "WATCHDOG_LOG_PATH", log_path)

        kp._log_group_timing(datetime(2026, 7, 19), "AAPL", "1d", "Base", 1.0, True)

        with open(log_path) as f:
            content = f.read()
        assert "cache=HIT" in content

    def test_never_raises_on_write_failure(self, monkeypatch):
        import kairos_papertrade as kp

        monkeypatch.setattr(kp, "WATCHDOG_LOG_PATH", "/nonexistent/\0bad/path.log")
        kp._log_group_timing(datetime(2026, 7, 19), "AAPL", "1d", "Base", 1.0, True)


class TestMakeGroupTimingCb:
    def test_logs_on_cache_miss_even_if_fast(self, monkeypatch, tmp_path):
        import kairos_papertrade as kp

        log_path = str(tmp_path / "papertrade_watchdog.log")
        monkeypatch.setattr(kp, "WATCHDOG_LOG_PATH", log_path)

        cb = kp._make_group_timing_cb(datetime(2026, 7, 19))
        cb("AAPL", "1d", "Base", 0.1, False)

        assert os.path.exists(log_path)

    def test_logs_on_slow_elapsed_even_if_cache_hit(self, monkeypatch, tmp_path):
        import kairos_papertrade as kp

        log_path = str(tmp_path / "papertrade_watchdog.log")
        monkeypatch.setattr(kp, "WATCHDOG_LOG_PATH", log_path)

        cb = kp._make_group_timing_cb(datetime(2026, 7, 19))
        cb("AAPL", "1d", "Base", kp._SLOW_GROUP_THRESHOLD_SECONDS + 1, True)

        assert os.path.exists(log_path)

    def test_silent_when_fast_and_cache_hit(self, monkeypatch, tmp_path):
        import kairos_papertrade as kp

        log_path = str(tmp_path / "papertrade_watchdog.log")
        monkeypatch.setattr(kp, "WATCHDOG_LOG_PATH", log_path)

        cb = kp._make_group_timing_cb(datetime(2026, 7, 19))
        cb("AAPL", "1d", "Base", 0.1, True)

        assert not os.path.exists(log_path)


# ============================================================================
# prewarm_prediction_cache
# ============================================================================

class TestPrewarmPredictionCache:
    def test_model_major_ordering(self, monkeypatch):
        import kairos_strategies

        calls = []

        def fake_predict_all_batch(data, model_path=None, tokenizer_path=None):
            calls.append((model_path, tuple(sorted(data.keys()))))
            return {}

        def fake_fetch_data_raw(sym, lookback, as_of=None):
            idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
            return pd.DataFrame({"close": [1.0] * lookback}, index=idx)

        monkeypatch.setattr(kairos_strategies, "predict_all_batch", fake_predict_all_batch)
        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", fake_fetch_data_raw)
        monkeypatch.setattr(kairos_strategies, "LOOKBACK", 10)

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(kp.sqlite3, "connect", lambda db_path: _FakeConn())

        # 2 groups; only "AAA,BBB" has an accepted finetuned model.
        groups = {
            ("AAA,BBB", "1d"): [{"assets": "AAA,BBB", "interval": "1d"}],
            ("CCC", "1d"): [{"assets": "CCC", "interval": "1d"}],
        }
        accepted_finetuned = {("AAA,BBB", "1d"): "repo/finetuned-ab"}

        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items",
                             lambda conn, intervals=None, include_all=False: [])
        monkeypatch.setattr(kp._kairos_signals_mod, "group_items", lambda rows: groups)
        monkeypatch.setattr(kp._kairos_signals_mod, "load_accepted_finetuned",
                             lambda conn: accepted_finetuned)

        base_now = datetime(2026, 7, 19, 0, 0)
        # 0.1 months * 30.44 / 1 day-per-step ~= 3.044 -> round() -> 3 dates.
        failures = kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        assert failures == []
        # Base sweep: 2 groups x 3 dates = 6 calls. Finetuned sweep: 1 group x 3 dates = 3 calls.
        assert len(calls) == 9

        base_positions = [i for i, c in enumerate(calls) if c[0] is None]
        finetuned_positions = [i for i, c in enumerate(calls) if c[0] is not None]
        assert len(base_positions) == 6
        assert len(finetuned_positions) == 3
        assert all(c[0] == "repo/finetuned-ab" for i, c in enumerate(calls) if i in finetuned_positions)

        # Model-major: every base call precedes every finetuned call.
        assert max(base_positions) < min(finetuned_positions)
        # The finetuned group's 3 calls are contiguous (not interleaved).
        assert finetuned_positions == list(range(min(finetuned_positions), min(finetuned_positions) + 3))

    def test_base_only_skips_finetuned_sweep(self, monkeypatch):
        import kairos_strategies

        calls = []

        def fake_predict_all_batch(data, model_path=None, tokenizer_path=None):
            calls.append(model_path)
            return {}

        def fake_fetch_data_raw(sym, lookback, as_of=None):
            idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
            return pd.DataFrame({"close": [1.0] * lookback}, index=idx)

        monkeypatch.setattr(kairos_strategies, "predict_all_batch", fake_predict_all_batch)
        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", fake_fetch_data_raw)
        monkeypatch.setattr(kairos_strategies, "LOOKBACK", 10)

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(kp.sqlite3, "connect", lambda db_path: _FakeConn())

        groups = {("AAA,BBB", "1d"): [{"assets": "AAA,BBB", "interval": "1d"}]}

        def _boom_load_accepted_finetuned(conn):
            raise AssertionError("load_accepted_finetuned should be skipped under base_only")

        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items",
                             lambda conn, intervals=None, include_all=False: [])
        monkeypatch.setattr(kp._kairos_signals_mod, "group_items", lambda rows: groups)
        monkeypatch.setattr(kp._kairos_signals_mod, "load_accepted_finetuned",
                             _boom_load_accepted_finetuned)

        base_now = datetime(2026, 7, 19, 0, 0)
        failures = kp.prewarm_prediction_cache(
            base_now, "1d", months_back=0.1, run_kwargs={"base_only": True},
        )

        assert failures == []
        assert calls == [None, None, None]

    def test_bad_date_does_not_abort_sweep(self, monkeypatch):
        import kairos_strategies

        calls = []
        fetch_attempts = {"n": 0}

        def flaky_fetch_data_raw(sym, lookback, as_of=None):
            fetch_attempts["n"] += 1
            if fetch_attempts["n"] == 2:
                raise RuntimeError("simulated fetch failure")
            idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
            return pd.DataFrame({"close": [1.0] * lookback}, index=idx)

        def fake_predict_all_batch(data, model_path=None, tokenizer_path=None):
            calls.append(model_path)
            return {}

        monkeypatch.setattr(kairos_strategies, "predict_all_batch", fake_predict_all_batch)
        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", flaky_fetch_data_raw)
        monkeypatch.setattr(kairos_strategies, "LOOKBACK", 10)

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(kp.sqlite3, "connect", lambda db_path: _FakeConn())

        groups = {("AAA", "1d"): [{"assets": "AAA", "interval": "1d"}]}
        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items",
                             lambda conn, intervals=None, include_all=False: [])
        monkeypatch.setattr(kp._kairos_signals_mod, "group_items", lambda rows: groups)
        monkeypatch.setattr(kp._kairos_signals_mod, "load_accepted_finetuned", lambda conn: {})

        base_now = datetime(2026, 7, 19, 0, 0)
        failures = kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        # 3 dates total; 1 fails, 2 succeed -- the failure must not abort the sweep.
        assert len(failures) == 1
        assert "simulated fetch failure" in failures[0]
        assert len(calls) == 2

    def _common_prewarm_mocks(self, monkeypatch, groups, accepted_finetuned=None):
        """Shared plumbing for the notify-gating tests below: fake
        fetch_data_raw/LOOKBACK, a fake sqlite3 connection, and
        group_items/load_work_items/load_accepted_finetuned stubs -- mirrors
        the setup already used by test_model_major_ordering etc."""
        import kairos_strategies

        def fake_fetch_data_raw(sym, lookback, as_of=None):
            idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
            return pd.DataFrame({"close": [1.0] * lookback}, index=idx)

        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", fake_fetch_data_raw)
        monkeypatch.setattr(kairos_strategies, "LOOKBACK", 10)

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(kp.sqlite3, "connect", lambda db_path: _FakeConn())
        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items",
                             lambda conn, intervals=None, include_all=False: [])
        monkeypatch.setattr(kp._kairos_signals_mod, "group_items", lambda rows: groups)
        monkeypatch.setattr(kp._kairos_signals_mod, "load_accepted_finetuned",
                             lambda conn: accepted_finetuned or {})

    def test_notify_skipped_when_sweep_fully_cached(self, monkeypatch):
        """Every (symbol, date) for both the base sweep and the finetuned
        group's sweep is already a shared-cache hit -> no _notify call for
        either unit."""
        import kairos_strategies

        groups = {("AAA,BBB", "1d"): [{"assets": "AAA,BBB", "interval": "1d"}]}
        accepted_finetuned = {("AAA,BBB", "1d"): "repo/finetuned-ab"}
        self._common_prewarm_mocks(monkeypatch, groups, accepted_finetuned)

        monkeypatch.setattr(kairos_strategies, "predict_all_batch",
                             lambda data, model_path=None, tokenizer_path=None: {})
        # Fully warm regardless of which model_path is being checked.
        monkeypatch.setattr(kairos_strategies, "is_batch_cached",
                             lambda data, model_path=None, pred_len=1: True)

        notify_calls = []
        monkeypatch.setattr(kp, "_notify", lambda text, enabled=True: notify_calls.append(text))

        base_now = datetime(2026, 7, 19, 0, 0)
        failures = kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        assert failures == []
        assert notify_calls == []

    def test_notify_fires_on_genuine_cache_miss_base_sweep(self, monkeypatch):
        """The base sweep has at least one genuine miss (finetuned model_path
        is fully warm) -> exactly one _notify call, labeled "Base", with the
        correct min/max date-range boundaries and date count."""
        import kairos_strategies

        groups = {("AAA,BBB", "1d"): [{"assets": "AAA,BBB", "interval": "1d"}]}
        accepted_finetuned = {("AAA,BBB", "1d"): "repo/finetuned-ab"}
        self._common_prewarm_mocks(monkeypatch, groups, accepted_finetuned)

        monkeypatch.setattr(kairos_strategies, "predict_all_batch",
                             lambda data, model_path=None, tokenizer_path=None: {})

        def fake_is_batch_cached(data, model_path=None, pred_len=1):
            # Base (model_path=None) is never cached; the finetuned group is
            # already fully warm.
            return model_path is not None

        monkeypatch.setattr(kairos_strategies, "is_batch_cached", fake_is_batch_cached)

        notify_calls = []
        monkeypatch.setattr(kp, "_notify", lambda text, enabled=True: notify_calls.append(text))

        base_now = datetime(2026, 7, 19, 0, 0)
        failures = kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        assert failures == []
        assert len(notify_calls) == 1
        msg = notify_calls[0]
        assert "Base" in msg
        assert "Finetuned" not in msg

        # dates = [base_now - i*1day for i in range(3)] (0.1 * 30.44 / 1 ~= 3.044 -> 3)
        expected_start = (base_now - timedelta(days=2)).strftime("%Y-%m-%d")
        expected_end = base_now.strftime("%Y-%m-%d")
        assert expected_start in msg
        assert expected_end in msg
        assert "3 dates" in msg

    def test_notify_fires_on_genuine_cache_miss_finetuned_sweep(self, monkeypatch):
        """The finetuned group's sweep has a genuine miss (base is fully
        warm) -> exactly one _notify call, labeled Finetuned(<assets>)."""
        import kairos_strategies

        groups = {("AAA,BBB", "1d"): [{"assets": "AAA,BBB", "interval": "1d"}]}
        accepted_finetuned = {("AAA,BBB", "1d"): "repo/finetuned-ab"}
        self._common_prewarm_mocks(monkeypatch, groups, accepted_finetuned)

        monkeypatch.setattr(kairos_strategies, "predict_all_batch",
                             lambda data, model_path=None, tokenizer_path=None: {})

        def fake_is_batch_cached(data, model_path=None, pred_len=1):
            # Base is already fully warm; the finetuned group is never cached.
            return model_path is None

        monkeypatch.setattr(kairos_strategies, "is_batch_cached", fake_is_batch_cached)

        notify_calls = []
        monkeypatch.setattr(kp, "_notify", lambda text, enabled=True: notify_calls.append(text))

        base_now = datetime(2026, 7, 19, 0, 0)
        failures = kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        assert failures == []
        assert len(notify_calls) == 1
        msg = notify_calls[0]
        assert "Finetuned(AAA,BBB)" in msg

        expected_start = (base_now - timedelta(days=2)).strftime("%Y-%m-%d")
        expected_end = base_now.strftime("%Y-%m-%d")
        assert expected_start in msg
        assert expected_end in msg
        assert "3 dates" in msg

    def test_notify_respects_notify_false(self, monkeypatch):
        """notify=False must reach _notify's own enabled= gate (mirrors
        generate_and_dedupe_reports' notify plumbing) -- assert _notify is
        invoked with enabled=False rather than skipped outright, matching
        the existing enabled= convention in this file."""
        import kairos_strategies

        groups = {("AAA,BBB", "1d"): [{"assets": "AAA,BBB", "interval": "1d"}]}
        self._common_prewarm_mocks(monkeypatch, groups, accepted_finetuned={})

        monkeypatch.setattr(kairos_strategies, "predict_all_batch",
                             lambda data, model_path=None, tokenizer_path=None: {})
        monkeypatch.setattr(kairos_strategies, "is_batch_cached",
                             lambda data, model_path=None, pred_len=1: False)

        notify_calls = []

        def fake_notify(text, enabled=True):
            notify_calls.append((text, enabled))

        monkeypatch.setattr(kp, "_notify", fake_notify)

        base_now = datetime(2026, 7, 19, 0, 0)
        kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={}, notify=False)

        assert len(notify_calls) == 1
        assert notify_calls[0][1] is False

    def test_notify_fires_before_model_load(self, monkeypatch):
        """The pre-load notification must be sent before predict_all_batch
        (the thing that actually triggers a real Kronos model load) is
        called for that sweep unit -- verified via a shared call-order list
        that both the _notify spy and the predict_all_batch fake append to."""
        import kairos_strategies

        order = []

        def fake_predict_all_batch(data, model_path=None, tokenizer_path=None):
            order.append(("predict", model_path))
            return {}

        def fake_notify(text, enabled=True):
            order.append(("notify", text))

        groups = {("AAA", "1d"): [{"assets": "AAA", "interval": "1d"}]}
        self._common_prewarm_mocks(monkeypatch, groups, accepted_finetuned={})

        monkeypatch.setattr(kairos_strategies, "predict_all_batch", fake_predict_all_batch)
        monkeypatch.setattr(kairos_strategies, "is_batch_cached",
                             lambda data, model_path=None, pred_len=1: False)
        monkeypatch.setattr(kp, "_notify", fake_notify)

        base_now = datetime(2026, 7, 19, 0, 0)
        failures = kp.prewarm_prediction_cache(
            base_now, "1d", months_back=0.1, run_kwargs={"base_only": True},
        )

        assert failures == []
        notify_positions = [i for i, entry in enumerate(order) if entry[0] == "notify"]
        predict_positions = [i for i, entry in enumerate(order) if entry[0] == "predict"]
        assert len(notify_positions) == 1
        assert len(predict_positions) == 3  # 3 dates, base_only
        assert notify_positions[0] < min(predict_positions)

    def test_tqdm_wrapping_does_not_change_calls_or_failures(self, monkeypatch):
        """Regression guard for the tqdm progress bars wrapping both passes
        (fetch+check, real predict) of both sweep units (base, finetuned
        group): predict_all_batch/_notify must still be invoked with the
        exact same arguments, and `failures` must still contain the exact
        same entries, as before the tqdm wrapping was added. tqdm's own
        console rendering is not asserted on -- only that it's a pure
        side-effect (progress reporting) with no bearing on the function's
        real behavior."""
        import kairos_strategies

        predict_calls = []
        notify_calls = []
        fetch_attempts = {"n": 0}

        def flaky_fetch_data_raw(sym, lookback, as_of=None):
            fetch_attempts["n"] += 1
            if fetch_attempts["n"] == 2:
                raise RuntimeError("simulated fetch failure")
            idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
            return pd.DataFrame({"close": [1.0] * lookback}, index=idx)

        def fake_predict_all_batch(data, model_path=None, tokenizer_path=None):
            predict_calls.append((model_path, tuple(sorted(data.keys()))))
            return {}

        def fake_is_batch_cached(data, model_path=None, pred_len=1):
            return False  # always a genuine miss -> notify fires for both units

        def fake_notify(text, enabled=True):
            notify_calls.append((text, enabled))

        groups = {("AAA,BBB", "1d"): [{"assets": "AAA,BBB", "interval": "1d"}]}
        accepted_finetuned = {("AAA,BBB", "1d"): "repo/finetuned-ab"}
        self._common_prewarm_mocks(monkeypatch, groups, accepted_finetuned)

        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", flaky_fetch_data_raw)
        monkeypatch.setattr(kairos_strategies, "predict_all_batch", fake_predict_all_batch)
        monkeypatch.setattr(kairos_strategies, "is_batch_cached", fake_is_batch_cached)
        monkeypatch.setattr(kp, "_notify", fake_notify)

        base_now = datetime(2026, 7, 19, 0, 0)
        failures = kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        # 3 dates; the group's 2nd fetch_data_raw call overall (2nd symbol
        # of the base sweep's 1st date) fails, aborting just that one date's
        # fetch dict comprehension -- the base sweep ends up with 1 failure
        # and 2 successfully-fetched dates; the finetuned sweep's own 3
        # fetches never land on the flaky counter's failing index again.
        assert len(failures) == 1
        assert "simulated fetch failure" in failures[0]

        # 2 successful base dates + 3 finetuned dates = 5 predict_all_batch calls.
        assert len(predict_calls) == 5
        base_predict = [c for c in predict_calls if c[0] is None]
        finetuned_predict = [c for c in predict_calls if c[0] == "repo/finetuned-ab"]
        assert len(base_predict) == 2
        assert len(finetuned_predict) == 3
        assert all(c[1] == ("AAA", "BBB") for c in predict_calls)

        # Both units had a genuine cache miss -> exactly one _notify per unit.
        assert len(notify_calls) == 2
        assert notify_calls[0][1] is True
        assert notify_calls[1][1] is True
        assert "Base" in notify_calls[0][0]
        assert "Finetuned(AAA,BBB)" in notify_calls[1][0]


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

    def test_pred_cache_defaults_true(self):
        args = _build_arg_parser().parse_args([])
        assert args.pred_cache is True

    def test_no_pred_cache_sets_pred_cache_false(self):
        args = _build_arg_parser().parse_args(["--no-pred-cache"])
        assert args.pred_cache is False


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


# ============================================================================
# _ensure_pred_cache_dir_env -- persistent cache-dir selection (extracted
# from main() so it's directly unit-testable without importing `phantom`)
# ============================================================================

class TestEnsurePredCacheDirEnv:
    def test_defaults_to_default_pred_cache_dir_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
        # Point DEFAULT_PRED_CACHE_DIR at a tmp_path-backed location so the
        # test never touches the real repo's data/ directory.
        fake_default = str(tmp_path / "predcache")
        monkeypatch.setattr(kp, "DEFAULT_PRED_CACHE_DIR", fake_default)

        result = _ensure_pred_cache_dir_env()

        assert result == fake_default
        assert os.environ["KAIROS_PRED_CACHE_DIR"] == fake_default
        assert os.path.isdir(fake_default)
        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)

    def test_does_not_delete_default_dir_or_its_contents(self, monkeypatch, tmp_path):
        # The whole point of persistence: unlike the old ephemeral tempdir
        # behavior, calling this (and by extension, a papertrade run) must
        # never remove the cache directory or what's in it.
        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
        fake_default = str(tmp_path / "predcache")
        monkeypatch.setattr(kp, "DEFAULT_PRED_CACHE_DIR", fake_default)

        _ensure_pred_cache_dir_env()
        marker = os.path.join(fake_default, "existing_entry.npz")
        with open(marker, "wb") as f:
            f.write(b"fake cached prediction data")

        # Call again (simulating a second invocation reusing the same dir).
        _ensure_pred_cache_dir_env()

        assert os.path.exists(marker), "persistent cache dir/contents must survive"
        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)

    def test_respects_preset_env_var_without_overwriting(self, monkeypatch, tmp_path):
        preset_dir = str(tmp_path / "caller_chosen_dir")
        os.makedirs(preset_dir, exist_ok=True)
        monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", preset_dir)
        fake_default = str(tmp_path / "predcache")
        monkeypatch.setattr(kp, "DEFAULT_PRED_CACHE_DIR", fake_default)

        result = _ensure_pred_cache_dir_env()

        assert result == preset_dir
        assert os.environ["KAIROS_PRED_CACHE_DIR"] == preset_dir
        # Must not have created/touched the default dir at all.
        assert not os.path.exists(fake_default)
        monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)


# ============================================================================
# _log_watchdog_snapshot -- best-effort forensic logging for the
# slow-iteration watchdog (freeze investigation; logging only, not a fix)
# ============================================================================

class TestReadSelfRssKb:
    def test_returns_positive_int_on_linux(self):
        rss = _read_self_rss_kb()
        assert rss is None or rss > 0

    def test_returns_none_when_proc_unreadable(self, monkeypatch):
        real_open = open

        def fake_open(path, *a, **kw):
            if path == "/proc/self/status":
                raise OSError("no /proc here")
            return real_open(path, *a, **kw)

        monkeypatch.setattr("builtins.open", fake_open)
        assert _read_self_rss_kb() is None


class TestLogWatchdogSnapshot:
    def test_appends_timestamped_entry_with_command_output(self, monkeypatch, tmp_path):
        log_path = str(tmp_path / "papertrade_watchdog.log")
        monkeypatch.setattr(kp, "WATCHDOG_LOG_PATH", log_path)

        def fake_run(cmd, capture_output, text, timeout):
            if cmd[0] == "free":
                return MagicMock(stdout="              total        used\nMem:           13Gi        8Gi\n")
            return MagicMock(stdout="memory.used,memory.total,utilization.gpu\n1000 MiB, 8000 MiB, 20 %\n")

        monkeypatch.setattr(kp.subprocess, "run", fake_run)

        _log_watchdog_snapshot("backtest 2026-07-28", 612.3)

        assert os.path.exists(log_path)
        with open(log_path) as f:
            content = f.read()
        assert "slow iteration" in content
        assert "backtest 2026-07-28" in content
        assert "elapsed=612.3s" in content
        assert "13Gi" in content
        assert "1000 MiB" in content
        assert f"pid={os.getpid()}" in content
        assert "self_rss=" in content

    def test_appends_without_clobbering_prior_entries(self, monkeypatch, tmp_path):
        log_path = str(tmp_path / "papertrade_watchdog.log")
        monkeypatch.setattr(kp, "WATCHDOG_LOG_PATH", log_path)
        monkeypatch.setattr(kp.subprocess, "run", lambda *a, **kw: MagicMock(stdout=""))

        _log_watchdog_snapshot("first", 400.0)
        _log_watchdog_snapshot("second", 500.0)

        with open(log_path) as f:
            content = f.read()
        assert "first" in content
        assert "second" in content

    def test_never_raises_when_subprocess_run_fails(self, monkeypatch, tmp_path):
        log_path = str(tmp_path / "papertrade_watchdog.log")
        monkeypatch.setattr(kp, "WATCHDOG_LOG_PATH", log_path)

        def boom(*a, **kw):
            raise FileNotFoundError("no such binary")

        monkeypatch.setattr(kp.subprocess, "run", boom)

        # Must not raise -- best-effort forensics, never blocks the backtest.
        _log_watchdog_snapshot("context", 301.0)

        assert os.path.exists(log_path)
        with open(log_path) as f:
            content = f.read()
        assert "free -h failed" in content
        assert "nvidia-smi failed" in content

    def test_creates_parent_directory_if_missing(self, monkeypatch, tmp_path):
        log_path = str(tmp_path / "nested" / "dir" / "papertrade_watchdog.log")
        monkeypatch.setattr(kp, "WATCHDOG_LOG_PATH", log_path)
        monkeypatch.setattr(kp.subprocess, "run", lambda *a, **kw: MagicMock(stdout=""))

        _log_watchdog_snapshot("context", 301.0)

        assert os.path.exists(log_path)
