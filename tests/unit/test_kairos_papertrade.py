import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import json
import sqlite3
from datetime import datetime, timedelta
from types import SimpleNamespace
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
    _format_start_sim_message,
    _format_finish_message,
    _format_crash_message,
    _ensure_pred_cache_dir_env,
    _raise_fd_limit,
    _log_watchdog_snapshot,
    _read_self_rss_kb,
    _ensure_mtm_daily_table,
    _insert_mtm_daily_row,
    compute_final_metrics,
    _fill_cash_delta,
    _close_cash_delta,
    _use_full_notional,
    _place_order_if_admitted,
    _place_batch_orders,
    _liquidate_position,
    compute_corrected_realized_pnl,
    DEFAULT_PRED_CACHE_DIR,
)
from allocation import AllocationConfig
from kairos_margin import load_margin_config
from kairos_mtm import DailySnapshot, OpenPositionView, compute_daily_snapshot, liquidation_check
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
    @pytest.fixture(autouse=True)
    def _isolated_seen_store(self, monkeypatch):
        # generate_and_dedupe_reports computes its seen-table name via
        # _make_report_hash, which reads the real pipeline_results.db --
        # stub it out so these unit tests stay hermetic. Also replace the
        # persistent SqliteDict with an in-memory dict so tests do not
        # collide with a real run's report_seen.db in the repo root.
        monkeypatch.setattr(
            kp, "_make_report_hash", lambda *a, **k: ("testhash", "legacy_testhash")
        )
        monkeypatch.setattr(kp, "_pick_seen_table", lambda *a, **k: "seen_v2_testhash")
        monkeypatch.setattr(
            kp, "SqliteDict", lambda filename, tablename, autocommit=True: {}
        )

    def test_returns_one_result_per_iter_now(self, tmp_path, monkeypatch):
        # The current implementation keys the seen map by iter_now, not by
        # parsed effective_dt, so distinct iter_now values survive even when
        # their reports share the same effective_dt header.
        calls = []

        def fake_run(now, intervals, return_rows, **kwargs):
            calls.append(now)
            # First two calls collapse to the same effective_dt (weekend dup).
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
        # 3 distinct iter_now values survive (no effective_dt de-dup).
        assert len(result) == 3
        # Sorted oldest-first by iter_now (stored as the tuple's first element).
        assert result[0][0] < result[1][0] < result[2][0]

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
        generate_and_dedupe_reports(base_now, "1d", months_back=0.1, run_kwargs={}, notify=True)

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
        generate_and_dedupe_reports(base_now, "1d", months_back=0.1, run_kwargs={}, notify=True)

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
        generate_and_dedupe_reports(base_now, "1d", months_back=0.1, run_kwargs={}, notify=False)

        assert len(calls) == 3
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

    def test_resume_skips_already_seen_dates(self, tmp_path, monkeypatch):
        # A second invocation sharing the same in-memory seen store must not
        # re-run dates already in the map -- only new, older dates get run().
        shared = {}
        monkeypatch.setattr(
            kp, "SqliteDict", lambda filename, tablename, autocommit=True: shared
        )

        calls, fake_run = self._fake_run_factory(tmp_path)
        monkeypatch.setattr(kp._kairos_signals_mod, "run", fake_run)

        base_now = datetime(2026, 7, 19, 0, 0)
        generate_and_dedupe_reports(base_now, "1d", months_back=0.1, run_kwargs={})
        assert len(calls) == 3

        second = generate_and_dedupe_reports(base_now, "1d", months_back=0.2, run_kwargs={})
        # 0.2 months ~= 6 iterations, 3 of which were already seen.
        assert len(calls) == 6
        assert len(second) == 6


# ============================================================================
# Report hash / table selection
# ============================================================================

class TestReportHashAndTableSelection:
    def test_make_report_hash_v2_includes_finetuned_paths(self, monkeypatch):
        import kairos_papertrade as kp

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: MagicMock())
        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items", lambda conn, intervals: [])
        monkeypatch.setattr(
            kp._kairos_signals_mod, "group_items", lambda rows: {("A,B", "1d"): []}
        )
        monkeypatch.setattr(
            kp._kairos_signals_mod, "load_accepted_finetuned",
            lambda conn: {("A,B", "1d"): "/models/v1"},
        )

        v2_a, legacy_a = kp._make_report_hash(datetime(2026, 7, 19), "1d", {})

        monkeypatch.setattr(
            kp._kairos_signals_mod, "load_accepted_finetuned",
            lambda conn: {("A,B", "1d"): "/models/v2"},
        )
        v2_b, legacy_b = kp._make_report_hash(datetime(2026, 7, 19), "1d", {})

        # v2 changes when the accepted finetuned model changes; legacy stays
        # identical so existing seen_<legacy> tables are still found.
        assert v2_a != v2_b
        assert legacy_a == legacy_b

    def test_make_report_hash_base_only_ignores_finetuned(self, monkeypatch):
        import kairos_papertrade as kp

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: MagicMock())
        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items", lambda conn, intervals: [])
        monkeypatch.setattr(
            kp._kairos_signals_mod, "group_items", lambda rows: {("A,B", "1d"): []}
        )
        monkeypatch.setattr(
            kp._kairos_signals_mod, "load_accepted_finetuned",
            lambda conn: {("A,B", "1d"): "/models/v1"},
        )

        v2_with, legacy_with = kp._make_report_hash(datetime(2026, 7, 19), "1d", {})
        v2_without, legacy_without = kp._make_report_hash(
            datetime(2026, 7, 19), "1d", {"base_only": True}
        )

        # base_only=True strips finetuned paths from v2.
        assert v2_with != v2_without
        assert legacy_with == legacy_without

    def test_pick_seen_table_prefers_populated_v2(self, tmp_path):
        import kairos_papertrade as kp
        db_path = str(tmp_path / "report_seen.db")
        # Both tables exist; v2 has rows, legacy is empty.
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE seen_v2_hash (key TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO seen_v2_hash VALUES ('x')")
        conn.execute("CREATE TABLE seen_hash (key TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()
        assert kp._pick_seen_table(db_path, "hash", "hash") == "seen_v2_hash"

    def test_pick_seen_table_falls_back_to_legacy(self, tmp_path):
        import kairos_papertrade as kp
        db_path = str(tmp_path / "report_seen.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE seen_v2_hash (key TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE seen_hash (key TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO seen_hash VALUES ('x')")
        conn.commit()
        conn.close()
        assert kp._pick_seen_table(db_path, "hash", "hash") == "seen_hash"

    def test_pick_seen_table_defaults_to_v2_when_empty(self, tmp_path):
        import kairos_papertrade as kp
        db_path = str(tmp_path / "report_seen.db")
        assert kp._pick_seen_table(db_path, "hash", "hash") == "seen_v2_hash"


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

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: _FakeConn())

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

    def test_data_refetched_not_retained_between_check_and_load_passes(self, monkeypatch):
        """Regression guard for the 2026-07-29 prewarm memory leak: a live
        run held every fetched (group, date) DataFrame in a list across the
        whole base sweep, growing RSS from ~2.4GB to 10.1GB in 18 minutes
        before a single new predict_all_batch call. The fix re-fetches data
        fresh in the load pass instead of reusing what the check pass
        fetched, so the load pass's fetches are always independent re-reads
        -- this test fails if someone reverts to threading fetched data from
        check straight into load. With no shared cache active, is_batch_cached
        is a miss on the very first checked entry, so the check pass stops
        after just that one date (2026-07-29 speed fix -- see
        prewarm_prediction_cache's docstring) and the load pass then covers
        the full 3-date x 2-symbol cross product on its own."""
        import kairos_strategies

        fetch_calls = []

        def counting_fetch_data_raw(sym, lookback, as_of=None):
            fetch_calls.append((sym, as_of))
            idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
            return pd.DataFrame({"close": [1.0] * lookback}, index=idx)

        monkeypatch.setattr(kairos_strategies, "predict_all_batch",
                             lambda data, model_path=None, tokenizer_path=None: {})
        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", counting_fetch_data_raw)
        monkeypatch.setattr(kairos_strategies, "LOOKBACK", 10)

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: _FakeConn())

        groups = {("AAA,BBB", "1d"): [{"assets": "AAA,BBB", "interval": "1d"}]}
        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items",
                             lambda conn, intervals=None, include_all=False: [])
        monkeypatch.setattr(kp._kairos_signals_mod, "group_items", lambda rows: groups)
        monkeypatch.setattr(kp._kairos_signals_mod, "load_accepted_finetuned", lambda conn: {})

        base_now = datetime(2026, 7, 19, 0, 0)
        # 0.1 months -> 3 dates; 1 group of 2 symbols -> 2 fetches/date/pass.
        failures = kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        assert failures == []
        # Check pass: 1 date x 2 symbols (stops after the first miss) = 2.
        # Load pass: full cross product, 3 dates x 2 symbols = 6.
        assert len(fetch_calls) == 8

    def test_gc_collect_called_periodically_during_long_check_pass(self, monkeypatch):
        """Regression guard for the 2026-07-29 leak: gc.collect() is
        otherwise only called on a model switch (_materialize_model), and
        the base sweep never switches models, so a long, fully-cached check
        pass (no misses -> runs to completion, see the early-exit speedup)
        would previously never trigger a single collection. With
        _PREWARM_GC_INTERVAL patched down to 3 and every entry a cache hit,
        2 groups x 3 dates = 6 check iterations must cross it at least once."""
        import kairos_strategies

        monkeypatch.setattr(kp, "_PREWARM_GC_INTERVAL", 3)
        gc_calls = {"n": 0}
        monkeypatch.setattr(kp.gc, "collect", lambda: gc_calls.__setitem__("n", gc_calls["n"] + 1))

        def fake_fetch_data_raw(sym, lookback, as_of=None):
            idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
            return pd.DataFrame({"close": [1.0] * lookback}, index=idx)

        monkeypatch.setattr(kairos_strategies, "predict_all_batch",
                             lambda data, model_path=None, tokenizer_path=None: {})
        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", fake_fetch_data_raw)
        monkeypatch.setattr(kairos_strategies, "is_batch_cached", lambda data, model_path=None, pred_len=1: True)
        monkeypatch.setattr(kairos_strategies, "LOOKBACK", 10)

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: _FakeConn())

        groups = {
            ("AAA", "1d"): [{"assets": "AAA", "interval": "1d"}],
            ("BBB", "1d"): [{"assets": "BBB", "interval": "1d"}],
        }
        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items",
                             lambda conn, intervals=None, include_all=False: [])
        monkeypatch.setattr(kp._kairos_signals_mod, "group_items", lambda rows: groups)
        monkeypatch.setattr(kp._kairos_signals_mod, "load_accepted_finetuned", lambda conn: {})

        base_now = datetime(2026, 7, 19, 0, 0)
        # 0.1 months -> 3 dates; 2 groups x 3 dates = 6 check iterations,
        # no misses -> runs to completion, load pass skipped entirely.
        kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        assert gc_calls["n"] >= 1

    def test_gc_collect_called_periodically_during_load_pass(self, monkeypatch):
        """Companion to the check-pass test above: once a miss triggers the
        load pass, that loop must also periodically collect -- it iterates
        the full cross product regardless of how much the check pass
        covered before stopping early."""
        import kairos_strategies

        monkeypatch.setattr(kp, "_PREWARM_GC_INTERVAL", 3)
        gc_calls = {"n": 0}
        monkeypatch.setattr(kp.gc, "collect", lambda: gc_calls.__setitem__("n", gc_calls["n"] + 1))

        def fake_fetch_data_raw(sym, lookback, as_of=None):
            idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
            return pd.DataFrame({"close": [1.0] * lookback}, index=idx)

        monkeypatch.setattr(kairos_strategies, "predict_all_batch",
                             lambda data, model_path=None, tokenizer_path=None: {})
        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", fake_fetch_data_raw)
        # Every entry a miss -> check pass stops after 1, load pass covers
        # the full 2 groups x 3 dates = 6 iterations.
        monkeypatch.setattr(kairos_strategies, "is_batch_cached", lambda data, model_path=None, pred_len=1: False)
        monkeypatch.setattr(kairos_strategies, "LOOKBACK", 10)

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: _FakeConn())

        groups = {
            ("AAA", "1d"): [{"assets": "AAA", "interval": "1d"}],
            ("BBB", "1d"): [{"assets": "BBB", "interval": "1d"}],
        }
        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items",
                             lambda conn, intervals=None, include_all=False: [])
        monkeypatch.setattr(kp._kairos_signals_mod, "group_items", lambda rows: groups)
        monkeypatch.setattr(kp._kairos_signals_mod, "load_accepted_finetuned", lambda conn: {})

        base_now = datetime(2026, 7, 19, 0, 0)
        kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        assert gc_calls["n"] >= 1

    def test_check_pass_stops_checking_at_first_miss(self, monkeypatch):
        """2026-07-29 speed request: the check pass has all the information
        it needs (needs_load=True) the moment ONE entry is a miss -- it must
        not keep calling is_batch_cached (or fetching) for every remaining
        entry in the unit. 5 groups x 3 dates = 15 possible check iterations;
        the very first is a miss, so at most 1 should ever be checked."""
        import kairos_strategies

        is_batch_cached_calls = []

        def fake_fetch_data_raw(sym, lookback, as_of=None):
            idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
            return pd.DataFrame({"close": [1.0] * lookback}, index=idx)

        def fake_is_batch_cached(data, model_path=None, pred_len=1):
            is_batch_cached_calls.append(model_path)
            return False  # every entry is a miss

        monkeypatch.setattr(kairos_strategies, "predict_all_batch",
                             lambda data, model_path=None, tokenizer_path=None: {})
        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", fake_fetch_data_raw)
        monkeypatch.setattr(kairos_strategies, "is_batch_cached", fake_is_batch_cached)
        monkeypatch.setattr(kairos_strategies, "LOOKBACK", 10)

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: _FakeConn())

        groups = {(f"SYM{i}", "1d"): [{"assets": f"SYM{i}", "interval": "1d"}] for i in range(5)}
        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items",
                             lambda conn, intervals=None, include_all=False: [])
        monkeypatch.setattr(kp._kairos_signals_mod, "group_items", lambda rows: groups)
        monkeypatch.setattr(kp._kairos_signals_mod, "load_accepted_finetuned", lambda conn: {})

        base_now = datetime(2026, 7, 19, 0, 0)
        failures = kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        assert failures == []
        assert len(is_batch_cached_calls) == 1  # stopped after the very first check

    def test_load_pass_skipped_and_logged_when_fully_cached(self, monkeypatch, capsys):
        """2026-07-29 speed request: if the check pass finds zero misses,
        the load pass must not run at all (no predict_all_batch calls), and
        the skip must be visible on the console."""
        import kairos_strategies

        predict_calls = []

        def fake_fetch_data_raw(sym, lookback, as_of=None):
            idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
            return pd.DataFrame({"close": [1.0] * lookback}, index=idx)

        def fake_predict_all_batch(data, model_path=None, tokenizer_path=None):
            predict_calls.append(model_path)
            return {}

        monkeypatch.setattr(kairos_strategies, "predict_all_batch", fake_predict_all_batch)
        monkeypatch.setattr(kairos_strategies, "fetch_data_raw", fake_fetch_data_raw)
        monkeypatch.setattr(kairos_strategies, "is_batch_cached", lambda data, model_path=None, pred_len=1: True)
        monkeypatch.setattr(kairos_strategies, "LOOKBACK", 10)

        class _FakeConn:
            def close(self):
                pass

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: _FakeConn())

        groups = {("AAA,BBB", "1d"): [{"assets": "AAA,BBB", "interval": "1d"}]}
        accepted_finetuned = {("AAA,BBB", "1d"): "repo/finetuned-ab"}
        monkeypatch.setattr(kp._kairos_signals_mod, "load_work_items",
                             lambda conn, intervals=None, include_all=False: [])
        monkeypatch.setattr(kp._kairos_signals_mod, "group_items", lambda rows: groups)
        monkeypatch.setattr(kp._kairos_signals_mod, "load_accepted_finetuned",
                             lambda conn: accepted_finetuned)

        base_now = datetime(2026, 7, 19, 0, 0)
        failures = kp.prewarm_prediction_cache(base_now, "1d", months_back=0.1, run_kwargs={})

        assert failures == []
        assert predict_calls == []  # load pass never ran for either unit

        out = capsys.readouterr().out
        assert "Prewarm load: Base skipped" in out
        assert "Prewarm load: Finetuned(AAA,BBB) skipped" in out

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

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: _FakeConn())

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

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: _FakeConn())

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

        monkeypatch.setattr(kp._kairos_signals_mod, "_connect_with_retry", lambda db_path: _FakeConn())
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
        # fetch dict comprehension during the CHECK pass -- recorded as 1
        # failure. Every entry is a miss, so the check pass stops after its
        # 2nd date (the 1st failed, the 2nd found the miss); the load pass
        # then covers the FULL 3-date cross product on its own (it doesn't
        # depend on what the check pass fetched), re-attempting date0's
        # fetch too -- which succeeds this time, since the flaky counter
        # only ever fails once. So all 3 base dates end up predicted despite
        # date0's transient check-pass failure.
        assert len(failures) == 1
        assert "simulated fetch failure" in failures[0]

        # 3 base dates + 3 finetuned dates = 6 predict_all_batch calls.
        assert len(predict_calls) == 6
        base_predict = [c for c in predict_calls if c[0] is None]
        finetuned_predict = [c for c in predict_calls if c[0] == "repo/finetuned-ab"]
        assert len(base_predict) == 3
        assert len(finetuned_predict) == 3
        assert all(c[1] == ("AAA", "BBB") for c in predict_calls)

        # Both units had a genuine cache miss -> exactly one _notify per unit.
        assert len(notify_calls) == 2
        assert notify_calls[0][1] is True
        assert notify_calls[1][1] is True
        assert "Base" in notify_calls[0][0]
        assert "Finetuned(AAA,BBB)" in notify_calls[1][0]


# ============================================================================
# _fill_cash_delta / _close_cash_delta -- corrected_cash round-trip arithmetic
# (regression for the E4-S09 follow-up bug: entry costs were debited at fill
# and never restored at close, short-changing corrected_cash by exactly EC
# on every closed trade)
# ============================================================================

class TestCorrectedCashFillCloseDelta:
    # entry_price=100, qty=10 -> entry_notional=1000, EC=2+1.5+1+0.5=5.
    # exit_price=110 -> gross_pnl=(110-100)*10=100. EXC (exit-side
    # commission+spread+slippage, never stored on the position row) = 3, so
    # phantom's stored realized_pnl = gross_pnl - (EC_no_fx + EXC)
    # = 100 - (4.5 + 3) = 92.5, and corrected_realized_pnl
    # = realized_pnl - fx = 92.5 - 0.5 = 92 = gross_pnl - EC - EXC.
    CLOSED_POSITION = {
        "entry_price": 100.0, "quantity": 10.0,
        "commission_entry": 2.0, "spread_cost": 1.5,
        "slippage_cost": 1.0, "fx_conversion_cost": 0.5,
        "realized_pnl": 92.5,
    }

    def test_fill_delta_is_full_notional_plus_entry_costs_debit(self):
        # -(1000 + 5) = -1005
        assert _fill_cash_delta(self.CLOSED_POSITION) == pytest.approx(-1005.0)

    def test_corrected_realized_pnl_sanity(self):
        assert compute_corrected_realized_pnl(self.CLOSED_POSITION) == pytest.approx(92.0)

    def test_close_delta_restores_entry_costs_and_adds_corrected_pnl(self):
        # 1000 + 5 (EC) + 92 (corrected_realized_pnl) = 1097
        assert _close_cash_delta(self.CLOSED_POSITION) == pytest.approx(1097.0)

    def test_fill_then_close_round_trip_matches_true_economic_pnl(self):
        # Net effect of fill + close must equal gross_pnl - EC - EXC = 100 - 5 - 3 = 92,
        # NOT gross_pnl - 2*EC - EXC = 87 (the bug: EC debited at fill, never restored).
        capital = 10000.0
        cash = capital
        cash += _fill_cash_delta(self.CLOSED_POSITION)
        cash += _close_cash_delta(self.CLOSED_POSITION)
        assert cash == pytest.approx(10092.0)
        assert cash - capital == pytest.approx(92.0)


# ============================================================================
# E4-S10 -- admission check gating + margin-aware cash debits
# ============================================================================

@pytest.fixture
def margin_cfg():
    """Default IBKR-style margin config fixture (same file production loads)."""
    return load_margin_config(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "margin_ibkr.yaml")
    )


class TestUseFullNotional:
    def test_max_leverage_one_is_always_full_notional(self, margin_cfg):
        # AAPL falls through to the default `equity_cfd` class (im_pct=20,
        # NOT spot) -- with leverage off this must still be full-notional,
        # or legacy (max_leverage=1.0) cash handling would silently change.
        assert _use_full_notional("AAPL", margin_cfg, max_leverage=1.0) is True
        assert _use_full_notional("BTC-USD", margin_cfg, max_leverage=1.0) is True

    def test_leveraged_spot_class_is_full_notional(self, margin_cfg):
        # crypto_spot (BTC-USD) has initial_margin_pct == 100 -> still full notional.
        assert _use_full_notional("BTC-USD", margin_cfg, max_leverage=2.0) is True

    def test_leveraged_marginable_class_excludes_notional(self, margin_cfg):
        # AAPL -> default equity_cfd, im_pct=20 < 100 -> margin-locked, not spent.
        assert _use_full_notional("AAPL", margin_cfg, max_leverage=2.0) is False


class TestFillCloseDeltaMarginAware:
    # Same fixture position as TestCorrectedCashFillCloseDelta, but exercised
    # with include_notional=False (marginable class under leverage).
    POSITION = {
        "entry_price": 100.0, "quantity": 10.0,
        "commission_entry": 2.0, "spread_cost": 1.5,
        "slippage_cost": 1.0, "fx_conversion_cost": 0.5,
        "realized_pnl": 92.5,
    }

    def test_fill_delta_excludes_notional_when_margin_locked(self):
        # -(0 + 5) = -5, no notional debited.
        assert _fill_cash_delta(self.POSITION, include_notional=False) == pytest.approx(-5.0)

    def test_close_delta_excludes_notional_when_margin_locked(self):
        # 0 + 5 (EC) + 92 (corrected_realized_pnl) = 97, no notional credited.
        assert _close_cash_delta(self.POSITION, include_notional=False) == pytest.approx(97.0)

    def test_fill_then_close_round_trip_still_equals_corrected_pnl(self):
        # The notional term cancels out either way -- net effect is exactly
        # corrected_realized_pnl (92), the true economic P&L when no cash
        # notional ever actually moved.
        capital = 10000.0
        cash = capital
        cash += _fill_cash_delta(self.POSITION, include_notional=False)
        cash += _close_cash_delta(self.POSITION, include_notional=False)
        assert cash - capital == pytest.approx(92.0)

    def test_legacy_default_matches_pre_e4_s10_behavior(self):
        # Pinned against the OLD (pre-E4-S10) formulas: fill = -(notional+EC),
        # close = notional+EC+corrected_pnl. Calling with no include_notional
        # kwarg (default True) must reproduce them byte-for-byte.
        entry_notional = self.POSITION["entry_price"] * self.POSITION["quantity"]
        ec = 2.0 + 1.5 + 1.0 + 0.5
        old_fill = -(entry_notional + ec)
        old_close = entry_notional + ec + 92.0
        assert _fill_cash_delta(self.POSITION) == pytest.approx(old_fill)
        assert _close_cash_delta(self.POSITION) == pytest.approx(old_close)


class TestPlaceOrderIfAdmitted:
    def _snapshot(self, equity, initial_margin_used, gross_notional=0.0):
        return DailySnapshot(
            date=datetime(2026, 8, 7).date(), cash=equity, unrealized_pnl=0.0,
            equity=equity, gross_notional=gross_notional,
            initial_margin_used=initial_margin_used, maintenance_margin_used=0.0,
            free_margin=equity - initial_margin_used,
            margin_utilization=(initial_margin_used / equity if equity > 0 else 0.0),
            financing_accrued_day=0.0, liquidations=0,
        )

    def test_accepted_order_is_placed(self, margin_cfg):
        client = MagicMock()
        alloc_config = AllocationConfig(max_leverage=2.0, margin_utilization_cap=0.8)
        # equity=1000, no margin used yet; a small order stays under the cap.
        snapshot = self._snapshot(equity=1000.0, initial_margin_used=0.0)
        order = MagicMock()

        placed = _place_order_if_admitted(
            client, "acct1", order, "AAPL", 100.0, datetime(2026, 8, 7),
            snapshot, margin_cfg, alloc_config,
        )

        assert placed is True
        client.orders.place.assert_called_once_with("acct1", order)

    def test_rejected_order_is_skipped_and_logged(self, margin_cfg, capsys):
        client = MagicMock()
        alloc_config = AllocationConfig(max_leverage=2.0, margin_utilization_cap=0.8)
        # equity=100, already fully margined out -- any new order breaches the cap.
        snapshot = self._snapshot(equity=100.0, initial_margin_used=80.0)
        order = MagicMock()

        placed = _place_order_if_admitted(
            client, "acct1", order, "AAPL", 500.0, datetime(2026, 8, 7),
            snapshot, margin_cfg, alloc_config,
        )

        assert placed is False
        client.orders.place.assert_not_called()
        assert "MARGIN_REJECTED" in capsys.readouterr().err

    def test_first_iteration_with_no_snapshot_skips_check(self, margin_cfg):
        client = MagicMock()
        alloc_config = AllocationConfig(max_leverage=2.0, margin_utilization_cap=0.8)
        order = MagicMock()

        placed = _place_order_if_admitted(
            client, "acct1", order, "AAPL", 1_000_000.0, datetime(2026, 8, 7),
            None, margin_cfg, alloc_config,
        )

        assert placed is True
        client.orders.place.assert_called_once_with("acct1", order)

    def test_max_leverage_one_is_always_admitted(self, margin_cfg):
        # admission_check itself no-ops for max_leverage<=1.0; confirm the
        # wiring here doesn't reject even a snapshot that would otherwise breach.
        client = MagicMock()
        alloc_config = AllocationConfig(max_leverage=1.0, margin_utilization_cap=0.8)
        snapshot = self._snapshot(equity=100.0, initial_margin_used=1000.0)
        order = MagicMock()

        placed = _place_order_if_admitted(
            client, "acct1", order, "AAPL", 500.0, datetime(2026, 8, 7),
            snapshot, margin_cfg, alloc_config,
        )

        assert placed is True
        client.orders.place.assert_called_once_with("acct1", order)


class TestPlaceBatchOrders:
    """Regression coverage for the same-day batch-admission gap: top_k
    defaults to 12, so several orders can be admitted in one iteration
    before `last_snapshot` next refreshes (once per day, at iteration end).
    Checking every order in the batch against that one static snapshot lets
    each pass individually while the batch together breaches
    margin_utilization_cap -- _place_batch_orders must thread a running
    initial_margin_used through the batch instead.
    """

    def _snapshot(self, equity, initial_margin_used):
        return DailySnapshot(
            date=datetime(2026, 8, 7).date(), cash=equity, unrealized_pnl=0.0,
            equity=equity, gross_notional=0.0,
            initial_margin_used=initial_margin_used, maintenance_margin_used=0.0,
            free_margin=equity - initial_margin_used,
            margin_utilization=(initial_margin_used / equity if equity > 0 else 0.0),
            financing_accrued_day=0.0, liquidations=0,
        )

    def test_second_order_in_batch_rejected_once_first_consumes_headroom(self, margin_cfg):
        # AAPL -> default equity_cfd class, initial_margin_pct=20.
        # equity=1000, cap=0.8 -> initial_margin_used may not exceed 800.
        # Each order alone locks 3000*0.20=600, which is <=800 in isolation
        # (i.e. checked against the static start-of-day snapshot ALONE, both
        # would incorrectly pass) -- but the two together lock 1200 > 800.
        client = MagicMock()
        alloc_config = AllocationConfig(max_leverage=2.0, margin_utilization_cap=0.8)
        snapshot = self._snapshot(equity=1000.0, initial_margin_used=0.0)
        order1, order2 = MagicMock(), MagicMock()
        order_requests = [(order1, "AAPL", 3000.0), (order2, "AAPL", 3000.0)]

        rejected = _place_batch_orders(
            client, "acct1", order_requests, datetime(2026, 8, 7),
            snapshot, margin_cfg, alloc_config,
        )

        assert rejected == 1
        client.orders.place.assert_called_once_with("acct1", order1)

    def test_batch_admits_up_to_the_running_cap_then_rejects_rest(self, margin_cfg):
        client = MagicMock()
        alloc_config = AllocationConfig(max_leverage=2.0, margin_utilization_cap=0.8)
        snapshot = self._snapshot(equity=1000.0, initial_margin_used=0.0)
        orders = [MagicMock() for _ in range(4)]
        # Each locks 300*0.20=60 margin; cap allows floor(800/60)=13, so all
        # 4 should be admitted here -- sanity check the running total doesn't
        # over-reject when there IS enough headroom for the whole batch.
        order_requests = [(o, "AAPL", 300.0) for o in orders]

        rejected = _place_batch_orders(
            client, "acct1", order_requests, datetime(2026, 8, 7),
            snapshot, margin_cfg, alloc_config,
        )

        assert rejected == 0
        assert client.orders.place.call_count == 4

    def test_no_snapshot_admits_whole_batch(self, margin_cfg):
        # First iteration of a run: no snapshot yet, nothing to check against.
        client = MagicMock()
        alloc_config = AllocationConfig(max_leverage=2.0, margin_utilization_cap=0.8)
        orders = [MagicMock(), MagicMock()]
        order_requests = [(o, "AAPL", 1_000_000.0) for o in orders]

        rejected = _place_batch_orders(
            client, "acct1", order_requests, datetime(2026, 8, 7),
            None, margin_cfg, alloc_config,
        )

        assert rejected == 0
        assert client.orders.place.call_count == 2


# ============================================================================
# kairos_mtm_daily table (schema + insert/read-back round trip)
# ============================================================================

class TestMtmDailyTable:
    def test_ensure_creates_table_idempotently(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "phantom.db"))
        try:
            _ensure_mtm_daily_table(conn)
            _ensure_mtm_daily_table(conn)  # must not raise on a second call

            cols = {row[1] for row in conn.execute("PRAGMA table_info(kairos_mtm_daily)")}
            assert cols == {
                "account_name", "date", "cash", "unrealized_pnl", "equity",
                "gross_notional", "initial_margin_used", "maintenance_margin_used",
                "free_margin", "margin_utilization", "financing_accrued_day",
                "financing_accrued_total", "liquidations",
            }
        finally:
            conn.close()

    def test_insert_and_read_back_row(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "phantom.db"))
        try:
            _ensure_mtm_daily_table(conn)
            snapshot = DailySnapshot(
                date=datetime(2026, 7, 1).date(), cash=150.0, unrealized_pnl=5.0,
                equity=155.0, gross_notional=1000.0, initial_margin_used=100.0,
                maintenance_margin_used=50.0, free_margin=55.0, margin_utilization=0.645,
                financing_accrued_day=-1.5, liquidations=0,
            )
            _insert_mtm_daily_row(conn, "acct1", snapshot, financing_accrued_total=-3.0)

            row = conn.execute(
                "SELECT account_name, date, cash, unrealized_pnl, equity, gross_notional, "
                "initial_margin_used, maintenance_margin_used, free_margin, margin_utilization, "
                "financing_accrued_day, financing_accrued_total, liquidations "
                "FROM kairos_mtm_daily WHERE account_name = ? AND date = ?",
                ("acct1", "2026-07-01"),
            ).fetchone()

            assert row == (
                "acct1", "2026-07-01", 150.0, 5.0, 155.0, 1000.0, 100.0, 50.0, 55.0,
                0.645, -1.5, -3.0, 0,
            )
        finally:
            conn.close()

    def test_insert_or_replace_on_rerun(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "phantom.db"))
        try:
            _ensure_mtm_daily_table(conn)
            snap1 = DailySnapshot(
                date=datetime(2026, 7, 1).date(), cash=150.0, unrealized_pnl=0.0,
                equity=150.0, gross_notional=0.0, initial_margin_used=0.0,
                maintenance_margin_used=0.0, free_margin=150.0, margin_utilization=0.0,
                financing_accrued_day=0.0, liquidations=0,
            )
            snap2 = DailySnapshot(
                date=datetime(2026, 7, 1).date(), cash=200.0, unrealized_pnl=0.0,
                equity=200.0, gross_notional=0.0, initial_margin_used=0.0,
                maintenance_margin_used=0.0, free_margin=200.0, margin_utilization=0.0,
                financing_accrued_day=0.0, liquidations=0,
            )
            _insert_mtm_daily_row(conn, "acct1", snap1, financing_accrued_total=0.0)
            _insert_mtm_daily_row(conn, "acct1", snap2, financing_accrued_total=0.0)  # re-run, same key

            rows = conn.execute(
                "SELECT cash FROM kairos_mtm_daily WHERE account_name = ? AND date = ?",
                ("acct1", "2026-07-01"),
            ).fetchall()
            assert rows == [(200.0,)]
        finally:
            conn.close()


# ============================================================================
# E4-S12 -- MTM metrics block
# ============================================================================

def _mtm_snapshot(day, cash, equity, margin_utilization, liquidations=0):
    return DailySnapshot(
        date=day, cash=cash, unrealized_pnl=equity - cash, equity=equity,
        gross_notional=0.0, initial_margin_used=0.0, maintenance_margin_used=0.0,
        free_margin=0.0, margin_utilization=margin_utilization,
        financing_accrued_day=0.0, liquidations=liquidations,
    )


class TestComputeFinalMetricsMtm:
    CAPITAL = 200.0

    def _ph_instance(self, conn):
        ph = MagicMock()
        ph._conn = conn
        ph.positions.list.return_value = []
        ph.accounts.get.return_value.cash = self.CAPITAL
        return ph

    def test_seven_mtm_keys_present_with_plausible_values(self, tmp_path):
        conn = sqlite3.connect(str(tmp_path / "phantom.db"))
        try:
            _ensure_mtm_daily_table(conn)
            _insert_mtm_daily_row(
                conn, "acct1",
                _mtm_snapshot(datetime(2026, 7, 1).date(), 200.0, 200.0, 0.10, liquidations=0),
                financing_accrued_total=-0.5,
            )
            _insert_mtm_daily_row(
                conn, "acct1",
                _mtm_snapshot(datetime(2026, 7, 2).date(), 190.0, 210.0, 0.55, liquidations=1),
                financing_accrued_total=-1.2,
            )
            _insert_mtm_daily_row(
                conn, "acct1",
                _mtm_snapshot(datetime(2026, 7, 3).date(), 220.0, 220.0, 0.20, liquidations=0),
                financing_accrued_total=-1.7,
            )

            ph = self._ph_instance(conn)
            metrics = compute_final_metrics(ph, 1, "acct1", self.CAPITAL, ruined=True)

            # final equity 220.0, capital 200.0 -> +10%
            assert metrics["mtm_total_return_pct"] == pytest.approx(10.0)
            # peak of 0.10 / 0.55 / 0.20 is 0.55, not the last row's value
            assert metrics["mtm_margin_utilization_peak"] == pytest.approx(0.55)
            # last row's cumulative total, NOT sum(-0.5, -1.2, -1.7)
            assert metrics["mtm_financing_total_eur"] == pytest.approx(-1.7)
            # 0 + 1 + 0
            assert metrics["mtm_liquidation_events"] == 1
            assert metrics["mtm_ruined"] is True
            assert isinstance(metrics["mtm_sharpe"], float)
            assert isinstance(metrics["mtm_max_drawdown_pct"], float)

            # existing closed-trade keys still present and unaffected (no closed positions)
            assert metrics["num_trades"] == 0
            assert metrics["total_profit_eur"] == 0.0
        finally:
            conn.close()

    def test_graceful_degradation_no_table(self, tmp_path):
        """A legacy/no-MTM run (kairos_mtm_daily never created for this DB) must
        still return all 7 mtm_* keys with sane defaults, not raise."""
        conn = sqlite3.connect(str(tmp_path / "phantom.db"))
        try:
            ph = self._ph_instance(conn)
            metrics = compute_final_metrics(ph, 1, "acct1", self.CAPITAL, ruined=False)

            assert metrics["mtm_total_return_pct"] == 0.0
            assert metrics["mtm_max_drawdown_pct"] == 0.0
            assert metrics["mtm_sharpe"] == 0.0
            assert metrics["mtm_margin_utilization_peak"] == 0.0
            assert metrics["mtm_financing_total_eur"] == 0.0
            assert metrics["mtm_liquidation_events"] == 0
            assert metrics["mtm_ruined"] is False
        finally:
            conn.close()

    def test_graceful_degradation_table_exists_no_rows_for_account(self, tmp_path):
        """Table exists (created for a different account) but has zero rows for
        THIS account -- same graceful-default contract as no table at all."""
        conn = sqlite3.connect(str(tmp_path / "phantom.db"))
        try:
            _ensure_mtm_daily_table(conn)
            _insert_mtm_daily_row(
                conn, "other_acct",
                _mtm_snapshot(datetime(2026, 7, 1).date(), 200.0, 200.0, 0.10),
                financing_accrued_total=0.0,
            )
            ph = self._ph_instance(conn)
            metrics = compute_final_metrics(ph, 1, "acct1", self.CAPITAL, ruined=False)

            assert metrics["mtm_liquidation_events"] == 0
            assert metrics["mtm_ruined"] is False
        finally:
            conn.close()


# ============================================================================
# E4-S11 -- liquidation execution
# ============================================================================

class TestLiquidatePosition:
    # entry_price=100, qty=10, long -> entry_notional=1000, EC=2+1.5+1+0.5=5.
    # close_price=80 -> gross_pnl=(80-100)*10=-200 (a losing position, the
    # realistic liquidation case). realized_pnl_to_store = gross_pnl - the
    # three non-fx entry costs = -200 -2 -1.5 -1 = -204.5 (NOT gross_pnl
    # itself -- see _liquidate_position's docstring). corrected_realized_pnl
    # = -204.5 - fx(0.5) = -205 = gross_pnl - EC. close_delta (include_notional
    # True) = notional(1000) + EC(5) + corrected_pnl(-205) = 800. Round trip:
    # fill(-1005) + close(800) = -205 = gross_pnl - EC - EXC(0), matching the
    # zero-exit-cost liquidation simplification.
    def _pos(self, **overrides):
        base = dict(
            id="pos1", ticker="AAPL", direction="long",
            entry_price=100.0, quantity=10.0,
            commission_entry=2.0, spread_cost=1.5,
            slippage_cost=1.0, fx_conversion_cost=0.5,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_cash_delta_matches_worked_example(self, margin_cfg):
        conn = MagicMock()
        pos = self._pos()

        delta = _liquidate_position(
            conn, pos, close_price=80.0, exit_dt=datetime(2026, 8, 7),
            margin_config=margin_cfg, max_leverage=1.0,
        )

        assert delta == pytest.approx(800.0)

    def test_short_position_cash_delta(self, margin_cfg):
        # short: gross_pnl = (entry-close)*qty = (100-120)*10 = -200 (loss on
        # a short that rallied against it) -- same magnitude as the long case
        # above but via the direction-aware formula, not a coincidence.
        conn = MagicMock()
        pos = self._pos(direction="short")

        delta = _liquidate_position(
            conn, pos, close_price=120.0, exit_dt=datetime(2026, 8, 7),
            margin_config=margin_cfg, max_leverage=1.0,
        )

        assert delta == pytest.approx(800.0)

    def test_writes_status_liquidated_and_margin_call_reason(self, margin_cfg):
        conn = MagicMock()
        cur = conn.cursor.return_value
        pos = self._pos()
        exit_dt = datetime(2026, 8, 7, 4, 0, 0)

        _liquidate_position(
            conn, pos, close_price=80.0, exit_dt=exit_dt,
            margin_config=margin_cfg, max_leverage=1.0,
        )

        order_null_call, position_update_call = cur.execute.call_args_list
        assert "orders" in order_null_call.args[0]
        assert order_null_call.args[1] == ("pos1",)

        sql, params = position_update_call.args
        assert "status = 'liquidated'" in sql
        assert "close_reason = 'margin_call'" in sql
        close_price, exit_datetime, realized_pnl, position_id = params
        assert close_price == 80.0
        assert exit_datetime == exit_dt.isoformat()
        assert realized_pnl == pytest.approx(-204.5)
        assert position_id == "pos1"
        conn.commit.assert_called_once()

    def test_margin_locked_class_excludes_notional(self, margin_cfg):
        # AAPL under leverage -> equity_cfd class, im_pct=20 < 100 -> margin
        # bookkeeping excludes notional on both sides, matching _use_full_notional.
        conn = MagicMock()
        pos = self._pos()

        delta = _liquidate_position(
            conn, pos, close_price=80.0, exit_dt=datetime(2026, 8, 7),
            margin_config=margin_cfg, max_leverage=2.0,
        )

        # 0 (no notional) + EC(5) + corrected_pnl(-205) = -200
        assert delta == pytest.approx(-200.0)

    def test_real_db_writes_and_nulls_order_fk(self, tmp_path):
        # Small fixture schema mirroring the phantom columns this function
        # touches -- proves the SQL actually runs against sqlite, not just
        # that the right strings were passed to a mock.
        conn = sqlite3.connect(str(tmp_path / "phantom.db"))
        try:
            conn.execute(
                "CREATE TABLE positions (id TEXT PRIMARY KEY, status TEXT, "
                "exit_price REAL, exit_datetime TEXT, realized_pnl REAL, close_reason TEXT)"
            )
            conn.execute("CREATE TABLE orders (id TEXT PRIMARY KEY, position_id TEXT)")
            conn.execute("INSERT INTO positions (id, status) VALUES ('pos1', 'open')")
            conn.execute("INSERT INTO orders (id, position_id) VALUES ('order1', 'pos1')")
            conn.commit()

            margin_cfg_local = load_margin_config(
                os.path.join(os.path.dirname(__file__), "..", "..", "config", "margin_ibkr.yaml")
            )
            pos = self._pos()
            exit_dt = datetime(2026, 8, 7)

            delta = _liquidate_position(
                conn, pos, close_price=80.0, exit_dt=exit_dt,
                margin_config=margin_cfg_local, max_leverage=1.0,
            )

            assert delta == pytest.approx(800.0)
            row = conn.execute(
                "SELECT status, exit_price, exit_datetime, realized_pnl, close_reason "
                "FROM positions WHERE id = 'pos1'"
            ).fetchone()
            assert row == ("liquidated", 80.0, exit_dt.isoformat(), pytest.approx(-204.5), "margin_call")

            order_row = conn.execute("SELECT position_id FROM orders WHERE id = 'order1'").fetchone()
            assert order_row == (None,)
        finally:
            conn.close()


class TestLiquidationPipeline:
    """End-to-end (no phantom/main() involved) exercise of the same sequence
    main()'s day loop performs: compute a snapshot, run liquidation_check,
    liquidate via _liquidate_position, recompute the snapshot from
    post-liquidation cash and the surviving positions -- proving the pieces
    compose the way E4-S11 wires them together."""

    def test_trigger_liquidates_and_recomputed_snapshot_excludes_ticker(self, margin_cfg):
        # A single AAPL long, deep underwater, with equity far below the
        # closeout_fraction * initial_margin_used trigger.
        pos_view = OpenPositionView(
            ticker="AAPL", direction="long", entry_price=100.0, quantity=100.0,
            entry_costs=5.0,
        )
        day_bars = {"AAPL": {"date": datetime(2026, 8, 7).date(), "close": 50.0}}
        # cash deliberately small so equity (cash + unrealized_pnl) is deeply negative.
        cash = 100.0
        snapshot = compute_daily_snapshot([pos_view], day_bars, cash, margin_cfg)
        assert snapshot.equity < 0  # sanity: unrealized_pnl = (50-100)*100 = -5000

        tickers_liquidated, _post_equity, ruined = liquidation_check(snapshot, [pos_view], margin_cfg)
        assert tickers_liquidated == ["AAPL"]
        assert ruined is True  # only position liquidated, equity still <= 0

        conn = MagicMock()
        phantom_pos = SimpleNamespace(
            id="pos1", ticker="AAPL", direction="long", entry_price=100.0, quantity=100.0,
            commission_entry=2.0, spread_cost=1.5, slippage_cost=1.0, fx_conversion_cost=0.5,
        )
        corrected_cash = cash + _liquidate_position(
            conn, phantom_pos, close_price=50.0, exit_dt=datetime(2026, 8, 7),
            margin_config=margin_cfg, max_leverage=1.0,
        )

        remaining = [p for p in [pos_view] if p.ticker not in tickers_liquidated]
        recomputed = compute_daily_snapshot(remaining, day_bars, corrected_cash, margin_cfg)
        recomputed = kp.replace(recomputed, liquidations=len(tickers_liquidated))

        assert remaining == []
        assert recomputed.liquidations == 1
        # No positions left -> unrealized_pnl is 0, equity == corrected cash.
        assert recomputed.unrealized_pnl == 0.0
        assert recomputed.equity == pytest.approx(corrected_cash)

    def test_below_trigger_no_liquidation(self, margin_cfg):
        # Healthy position: equity comfortably above closeout_fraction * IM.
        pos_view = OpenPositionView(
            ticker="AAPL", direction="long", entry_price=100.0, quantity=10.0,
            entry_costs=5.0,
        )
        day_bars = {"AAPL": {"date": datetime(2026, 8, 7).date(), "close": 101.0}}
        snapshot = compute_daily_snapshot([pos_view], day_bars, cash=10000.0, cfg=margin_cfg)

        tickers_liquidated, post_equity, ruined = liquidation_check(snapshot, [pos_view], margin_cfg)

        assert tickers_liquidated == []
        assert ruined is False
        assert post_equity == snapshot.equity


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

    def test_margin_config_default_is_config_margin_ibkr_yaml(self):
        args = _build_arg_parser().parse_args([])
        assert args.margin_config == "config/margin_ibkr.yaml"

    def test_margin_config_accepts_custom_path(self):
        args = _build_arg_parser().parse_args(["--margin-config", "/path/to/custom.yaml"])
        assert args.margin_config == "/path/to/custom.yaml"

    def test_max_leverage_default_is_1_0(self):
        args = _build_arg_parser().parse_args([])
        assert args.max_leverage == 1.0

    def test_max_leverage_accepts_float(self):
        args = _build_arg_parser().parse_args(["--max-leverage", "2.5"])
        assert args.max_leverage == 2.5

    def test_margin_utilization_default_is_0_8(self):
        args = _build_arg_parser().parse_args([])
        assert args.margin_utilization == 0.8

    def test_margin_utilization_accepts_float(self):
        args = _build_arg_parser().parse_args(["--margin-utilization", "0.6"])
        assert args.margin_utilization == 0.6


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

    def test_contains_margin_leverage_params(self):
        args = _build_arg_parser().parse_args(
            ["--max-leverage", "2.0", "--margin-utilization", "0.7"]
        )
        base_now = datetime(2026, 7, 27, 12, 30)
        msg = _format_start_message(base_now, args)
        assert "max_leverage=2.0" in msg
        assert "margin_utilization=0.7" in msg


class TestFormatStartSimMessage:
    def test_contains_emoji_and_key_params(self):
        args = _build_arg_parser().parse_args(
            ["--interval", "1d", "--months-back", "3", "--top-n", "5",
             "--capital", "500", "--broker", "IBKR"]
        )
        base_now = datetime(2026, 7, 27, 12, 30)
        msg = _format_start_sim_message(base_now, args)
        assert msg.startswith("🟢")
        assert "simulating" in msg
        assert "2026-07-27 12:30" in msg
        assert "interval=1d" in msg
        assert "months_back=3.0" in msg
        assert "top_n=5" in msg
        assert "capital=500.0" in msg
        assert "broker=IBKR" in msg

    def test_contains_margin_leverage_params(self):
        args = _build_arg_parser().parse_args(
            ["--max-leverage", "2.0", "--margin-utilization", "0.7"]
        )
        base_now = datetime(2026, 7, 27, 12, 30)
        msg = _format_start_sim_message(base_now, args)
        assert "max_leverage=2.0" in msg
        assert "margin_utilization=0.7" in msg


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
# _raise_fd_limit -- regression guard for the 2026-07-29 "Too many open
# files" crash (OSError, Errno 24) hours into a 6-month run
# ============================================================================

class TestRaiseFdLimit:
    def test_raises_soft_to_hard_when_below(self, monkeypatch, capsys):
        monkeypatch.setattr(kp.resource, "getrlimit", lambda which: (1024, 1048576))
        calls = []
        monkeypatch.setattr(kp.resource, "setrlimit", lambda which, limits: calls.append(limits))

        _raise_fd_limit()

        assert calls == [(1048576, 1048576)]
        assert "1024 -> 1048576" in capsys.readouterr().err

    def test_noop_when_already_at_hard_limit(self, monkeypatch):
        monkeypatch.setattr(kp.resource, "getrlimit", lambda which: (1048576, 1048576))
        calls = []
        monkeypatch.setattr(kp.resource, "setrlimit", lambda which, limits: calls.append(limits))

        _raise_fd_limit()

        assert calls == []

    def test_noop_when_hard_limit_is_infinity(self, monkeypatch):
        monkeypatch.setattr(kp.resource, "getrlimit", lambda which: (1024, kp.resource.RLIM_INFINITY))
        calls = []
        monkeypatch.setattr(kp.resource, "setrlimit", lambda which, limits: calls.append(limits))

        _raise_fd_limit()

        assert calls == []

    def test_never_raises_when_setrlimit_fails(self, monkeypatch, capsys):
        monkeypatch.setattr(kp.resource, "getrlimit", lambda which: (1024, 1048576))

        def boom(which, limits):
            raise OSError("not permitted in this sandbox")

        monkeypatch.setattr(kp.resource, "setrlimit", boom)

        _raise_fd_limit()  # must not raise

        assert "could not raise open-file limit" in capsys.readouterr().err


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
