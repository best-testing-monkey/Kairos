"""Tests for scripts/run_model_parallel.py.

No GPU/model download needed: enumerate_backtest_dates() is pure pandas date
math, and the orchestrator-parity test drives the real KairosOrchestrator
with a stub batch_predict_fn that never touches a real model (config.
no_prediction stays False so _run_day routes through
multi_predictor.predict_all(), but the stub returns {} immediately, so no
strategy ever evaluates). select_prioritized_groups()/_chunked() are pure
DB/list logic.
"""
import json
import os
import sys
import threading
import time

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

import kairos_pipeline as kp
from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig

import run_model_parallel as rmp


def _frame(start, n, freq="D"):
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {"open": 1.0, "high": 1.01, "low": 0.99, "close": 1.0, "volume": 1000.0},
        index=idx,
    )


# ============================================================================
# enumerate_backtest_dates parity with KairosOrchestrator.run_backtest
# ============================================================================

def test_enumeration_matches_real_orchestrator_run_backtest():
    """Drives the real orchestrator (strategy/kairos_orchestrator.py:925-961)
    with a stub batch_predict_fn that records exactly the histories it was
    called with, then asserts enumerate_backtest_dates() produces the same
    (date, {symbol: last_bar_timestamp}) sequence. Staggered start dates so
    the `mask.sum() < lookback` skip actually gets exercised, not just a
    trivial equal-length case."""
    data_dict = {
        "AAA": _frame("2024-01-01", 30),
        "BBB": _frame("2024-01-10", 25),  # starts late -> some early dates skip BBB
    }
    lookback = 5

    calls = []

    def stub_batch_predict(assets, model_path=None, tokenizer_path=None):
        calls.append({sym: df.index[-1] for sym, df in assets.items()})
        return {}

    orch = KairosOrchestrator(
        predict_fn=lambda *a, **kw: [],
        assets=list(data_dict),
        config=OrchestratorConfig(no_prediction=False),
        batch_predict_fn=stub_batch_predict,
    )
    orch.run_backtest(data_dict, lookback=lookback)

    mine = [
        {sym: df.index[-1] for sym, df in histories.items()}
        for _date, histories in rmp.enumerate_backtest_dates(data_dict, lookback)
    ]

    assert mine == calls
    assert len(mine) > 0
    # sanity: BBB really was excluded on at least one early date
    assert any("BBB" not in call for call in mine)
    assert any("BBB" in call for call in mine)


def test_enumeration_empty_when_no_dates_pass_lookback():
    data_dict = {"AAA": _frame("2024-01-01", 3)}
    assert list(rmp.enumerate_backtest_dates(data_dict, lookback=10)) == []


# ============================================================================
# Chunking
# ============================================================================

def test_chunked_splits_evenly():
    chunks = list(rmp._chunked(list(range(10)), 4))
    assert chunks == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


def test_chunked_empty_input():
    assert list(rmp._chunked([], 4)) == []


def test_chunked_larger_than_input():
    assert list(rmp._chunked([1, 2], 10)) == [[1, 2]]


# ============================================================================
# select_prioritized_groups: skip-done / require-stage / require-since / ordering
# ============================================================================

@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "pipeline_test.db")
    conn = kp.get_connection(db_path)
    yield conn
    conn.close()


def _oracle(conn, assets, sharpe, backtest_period="6m"):
    run_id = kp.start_run(conn, "oracle", "1d", {})
    kp.insert_oracle_row(conn, run_id, {
        "stage": "oracle", "strategy_name": "s1", "sharpe": sharpe,
        "signal_count": 10, "win_rate": 0.5, "avg_pnl_per_trade": 0.01,
        "assets": assets, "interval": "1d", "backtest_period": backtest_period,
    })
    conn.commit()


def _model_row(conn, assets, stage, model_path, backtest_period="6m", timestamp=None):
    run_id = kp.start_run(conn, stage, "1d", {})
    if timestamp is not None:
        conn.execute("UPDATE runs SET timestamp=? WHERE run_id=?", (timestamp, run_id))
    kp.insert_model_row(conn, run_id, {
        "stage": stage, "strategy_name": "s1", "sharpe": 0.1,
        "signal_count": 5, "win_rate": 0.5, "avg_pnl_per_trade": 0.01,
        "assets": assets, "interval": "1d", "backtest_period": backtest_period,
        "model_path": model_path,
    })
    conn.commit()
    return run_id


def test_skips_group_already_done_for_stage_and_model(db):
    _model_row(db, "AAA,BBB", "small", "NeoQuasar/Kronos-small")
    groups = [(1, ["AAA", "BBB"]), (2, ["CCC"])]
    out = rmp.select_prioritized_groups(db, groups, "small", "NeoQuasar/Kronos-small")
    assert [g[0] for g in out] == [2]


def test_does_not_skip_same_group_for_different_model_path(db):
    _model_row(db, "AAA,BBB", "small", "NeoQuasar/Kronos-small")
    groups = [(1, ["AAA", "BBB"])]
    out = rmp.select_prioritized_groups(db, groups, "small", "NeoQuasar/Kronos-mini")
    assert [g[0] for g in out] == [1]


def test_require_stage_filters_to_paired_groups(db):
    _model_row(db, "AAA,BBB", "base", None)
    groups = [(1, ["AAA", "BBB"]), (2, ["CCC"])]
    out = rmp.select_prioritized_groups(db, groups, "small", "NeoQuasar/Kronos-small",
                                         require_stage="base")
    assert [g[0] for g in out] == [1]


def test_require_since_excludes_stale_paired_row(db):
    _model_row(db, "AAA,BBB", "base", None, timestamp="2026-01-01T00:00:00")
    groups = [(1, ["AAA", "BBB"])]
    out = rmp.select_prioritized_groups(db, groups, "small", "NeoQuasar/Kronos-small",
                                         require_stage="base", require_since="2026-06-01")
    assert out == []


def test_require_since_keeps_fresh_paired_row(db):
    _model_row(db, "AAA,BBB", "base", None, timestamp="2026-08-01T00:00:00")
    groups = [(1, ["AAA", "BBB"])]
    out = rmp.select_prioritized_groups(db, groups, "small", "NeoQuasar/Kronos-small",
                                         require_stage="base", require_since="2026-06-01")
    assert [g[0] for g in out] == [1]


def test_orders_by_oracle_sharpe_descending_unranked_last(db):
    _oracle(db, "AAA", sharpe=1.0)
    _oracle(db, "BBB", sharpe=5.0)
    groups = [(1, ["AAA"]), (2, ["BBB"]), (3, ["CCC"])]  # CCC has no oracle row
    out = rmp.select_prioritized_groups(db, groups, "small", None)
    assert [g[0] for g in out] == [2, 1, 3]


# ============================================================================
# --pipeline: run_pipelined() scheduling (no GPU/DB -- plain stub callables)
# ============================================================================

def test_pipelined_visits_each_chunk_once_in_order_with_real_overlap():
    """Sleep-based stub phase1/phase2 record wall-clock timestamps per event.
    Asserts: every chunk visited exactly once in order; each chunk's phase2
    starts only after its OWN phase1 finished; and chunk[i+1]'s phase1
    actually starts before chunk[i]'s phase2 finishes (real overlap, not
    just "both phases ran")."""
    lock = threading.Lock()
    times = {}

    def _log(label, chunk):
        with lock:
            times[(label, tuple(chunk))] = time.monotonic()

    def phase1(chunk):
        _log("phase1_start", chunk)
        time.sleep(0.05)
        _log("phase1_end", chunk)
        return ("p1", tuple(chunk))

    def phase2(chunk):
        _log("phase2_start", chunk)
        time.sleep(0.05)
        _log("phase2_end", chunk)
        return ("p2", tuple(chunk))

    chunks = [[1, 2], [3, 4], [5, 6]]
    out = list(rmp.run_pipelined(chunks, phase1, phase2))

    assert [c for c, *_ in out] == chunks  # every chunk, exactly once, in order

    for c in chunks:
        ct = tuple(c)
        assert times[("phase1_end", ct)] <= times[("phase2_start", ct)]

    # real overlap: next chunk's prewarm starts before this chunk's replay ends
    assert times[("phase1_start", (3, 4))] < times[("phase2_end", (1, 2))]
    assert times[("phase1_start", (5, 6))] < times[("phase2_end", (3, 4))]


def test_pipelined_prefetch_failure_falls_back_and_continues():
    calls = []
    failed_once = {"done": False}

    def phase1(chunk):
        calls.append(("phase1", tuple(chunk)))
        if chunk == [3, 4] and not failed_once["done"]:
            failed_once["done"] = True
            raise RuntimeError("boom")
        return ("p1", tuple(chunk))

    def phase2(chunk):
        calls.append(("phase2", tuple(chunk)))
        return ("p2", tuple(chunk))

    seen_fail = []

    def on_fail(chunk, exc):
        seen_fail.append((tuple(chunk), str(exc)))

    chunks = [[1, 2], [3, 4], [5, 6]]
    out = list(rmp.run_pipelined(chunks, phase1, phase2, on_prefetch_fail=on_fail))

    assert [c for c, *_ in out] == chunks  # run continues past the failure
    assert seen_fail == [((3, 4), "boom")]
    # [3,4]'s phase1 ran twice: the failed background prefetch, then the inline fallback
    assert calls.count(("phase1", (3, 4))) == 2

    fell_back_by_chunk = {tuple(c): fb for c, _p1, _p2, _w, _s, _wall, fb in out}
    assert fell_back_by_chunk == {(1, 2): False, (3, 4): True, (5, 6): False}


def test_parse_args_defaults_unchanged_pipeline_off_gpu_workers_int(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_model_parallel.py"])
    args = rmp._parse_args()
    assert args.pipeline is False
    assert args.gpu_workers == 1
    assert args.gpu_workers_max == 4


# ============================================================================
# --gpu-workers auto: decide_gpu_workers() pure sizing rule
# ============================================================================

def test_decide_gpu_workers_vram_bound():
    n = rmp.decide_gpu_workers(free_vram_mib=5000, avail_ram_mib=1_000_000,
                                per_worker_vram_mib=2200, per_worker_rss_mib=1100, max_workers=10)
    assert n == 2  # floor((5000-512)/2200) = 2, RAM/max not binding


def test_decide_gpu_workers_ram_bound():
    n = rmp.decide_gpu_workers(free_vram_mib=1_000_000, avail_ram_mib=5000,
                                per_worker_vram_mib=2200, per_worker_rss_mib=1100, max_workers=10)
    assert n == 3  # floor((5000-1024)/1100) = 3, VRAM/max not binding


def test_decide_gpu_workers_max_bound():
    n = rmp.decide_gpu_workers(free_vram_mib=1_000_000, avail_ram_mib=1_000_000,
                                per_worker_vram_mib=100, per_worker_rss_mib=100, max_workers=4)
    assert n == 4


def test_decide_gpu_workers_never_below_one():
    n = rmp.decide_gpu_workers(free_vram_mib=100, avail_ram_mib=100,
                                per_worker_vram_mib=2200, per_worker_rss_mib=1100, max_workers=4)
    assert n == 1


def test_decide_gpu_workers_pipelined_cpu_workers_running_reduces_answer():
    without = rmp.decide_gpu_workers(free_vram_mib=1_000_000, avail_ram_mib=8000,
                                      per_worker_vram_mib=2200, per_worker_rss_mib=1100,
                                      max_workers=10, cpu_workers_running=0)
    with_phase2 = rmp.decide_gpu_workers(free_vram_mib=1_000_000, avail_ram_mib=8000,
                                          per_worker_vram_mib=2200, per_worker_rss_mib=1100,
                                          max_workers=10, cpu_workers_running=6)
    assert with_phase2 < without


# ============================================================================
# --gpu-workers auto: _resolve_gpu_workers_for_chunk() (injected probes)
# ============================================================================

def test_resolve_gpu_workers_fixed_int_passes_through():
    n, log = rmp._resolve_gpu_workers_for_chunk(3, 4, 2200, 1100)
    assert n == 3
    assert "fixed" in log


def test_resolve_gpu_workers_probe_failure_falls_back_to_one_without_raising():
    """Both torch+CUDA and nvidia-smi unavailable -> probe returns None. Must
    yield 1, not raise. Injects the probe rather than monkeypatching
    subprocess."""
    n, log = rmp._resolve_gpu_workers_for_chunk(
        "auto", 4, 2200, 1100, probe_vram_fn=lambda: None,
    )
    assert n == 1
    assert "no GPU probe available" in log


def test_resolve_gpu_workers_auto_matches_worked_example():
    """Mirrors the exact numbers from the module's own auto-sizing log format."""
    n, log = rmp._resolve_gpu_workers_for_chunk(
        "auto", 4, 2200, 1100, probe_vram_fn=lambda: 5210, probe_ram_mib_fn=lambda: 8100,
    )
    assert n == 2
    assert "vram_free=5210MiB -> 2" in log
    assert "ram_avail=8100MiB -> 6" in log
    assert "max=4" in log


# ============================================================================
# VRAM calibration bug fix: _phase1_with_vram_calibration() actually moves
# per_worker_vram_mib, using per-process usage (_probe_own_vram_mib) instead
# of a whole-GPU memory.free before/after delta that a sibling session's own
# GPU usage can swamp. See _probe_own_vram_mib()'s docstring for the root
# cause this replaces.
# ============================================================================

def test_probe_own_vram_mib_sums_only_the_given_pids(monkeypatch):
    """Direct per-pid attribution: a third pid present in nvidia-smi's
    output (a sibling session/display server) must not be counted."""
    csv = "111, 560\n222, 560\n999, 4000\n"

    def fake_run(cmd, **kwargs):
        assert "--query-compute-apps=pid,used_memory" in cmd
        class R:
            stdout = csv
        return R()

    monkeypatch.setattr(rmp.subprocess, "run", fake_run)
    assert rmp._probe_own_vram_mib([111, 222]) == 1120.0


def test_probe_own_vram_mib_no_pids_given_is_zero_not_none():
    assert rmp._probe_own_vram_mib([]) == 0.0


def test_probe_own_vram_mib_none_of_our_pids_present_is_zero_not_none(monkeypatch):
    monkeypatch.setattr(
        rmp.subprocess, "run",
        lambda cmd, **kw: type("R", (), {"stdout": "999, 4000\n"})(),
    )
    assert rmp._probe_own_vram_mib([111]) == 0.0


def test_probe_own_vram_mib_nvidia_smi_failure_is_none(monkeypatch):
    def raise_(*a, **kw):
        raise FileNotFoundError("no nvidia-smi")
    monkeypatch.setattr(rmp.subprocess, "run", raise_)
    assert rmp._probe_own_vram_mib([111]) is None


def test_phase1_calibration_moves_estimate_from_a_known_per_worker_peak():
    """The regression test for the actual bug: feed a fake per-process VRAM
    sampler reporting a known peak for our own pids, and assert the NEXT
    chunk's decide_gpu_workers() input reflects it, not the DEFAULT seed --
    this is exactly what was broken (evidence: 7 straight chunks on a live
    run all showed vram_allows computed from the untouched 2200 seed)."""
    gpu_workers = 2
    fake_own_usage = {"value": 0.0}

    def fake_probe_own_vram(pids):
        # Simulate the pool having spawned 2 workers, each holding ~560MiB,
        # once run_phase1_fn has "started" the pool.
        return fake_own_usage["value"]

    def run_phase1_fn(pids_out):
        pids_out.extend([111, 222])
        fake_own_usage["value"] = 1120.0  # 2 workers x 560MiB, sampled mid-run
        time.sleep(0.05)  # give the sampler thread (interval=0.01 below) a chance to poll
        return "phase1-result"

    result, new_vram, new_rss = rmp._phase1_with_vram_calibration(
        gpu_workers, rmp.DEFAULT_PER_WORKER_VRAM_MIB, rmp.DEFAULT_PER_WORKER_RSS_MIB,
        auto=True, run_phase1_fn=run_phase1_fn, probe_own_vram_fn=fake_probe_own_vram,
        sample_interval=0.01,
    )

    assert result == "phase1-result"
    assert new_vram == pytest.approx(560.0)  # 1120 / 2 workers, not stuck at the 2200 seed
    assert new_vram != rmp.DEFAULT_PER_WORKER_VRAM_MIB

    # And the next chunk's auto-sizing decision actually reflects the update.
    n = rmp.decide_gpu_workers(free_vram_mib=5614, avail_ram_mib=1_000_000,
                                per_worker_vram_mib=new_vram, per_worker_rss_mib=new_rss,
                                max_workers=4)
    assert n == 4  # floor((5614-512)/560) = 9, capped at max_workers=4 -- NOT the old-bug's 2


def test_phase1_calibration_no_usage_seen_keeps_seed():
    """If our own pids never show any VRAM usage (e.g. every date in the
    chunk was already a cache hit, so predict_all_batch/model load never
    ran), the estimate must stay at whatever it was -- 0.0 usage is a valid
    reading, not something to divide into a bogus near-zero per-worker mib."""
    result, new_vram, _ = rmp._phase1_with_vram_calibration(
        2, rmp.DEFAULT_PER_WORKER_VRAM_MIB, rmp.DEFAULT_PER_WORKER_RSS_MIB,
        auto=True, run_phase1_fn=lambda pids: pids.extend([111, 222]) or "r",
        probe_own_vram_fn=lambda pids: 0.0,
    )
    assert new_vram == rmp.DEFAULT_PER_WORKER_VRAM_MIB


def test_phase1_calibration_auto_off_skips_sampling_entirely():
    calls = []
    result, new_vram, new_rss = rmp._phase1_with_vram_calibration(
        2, rmp.DEFAULT_PER_WORKER_VRAM_MIB, rmp.DEFAULT_PER_WORKER_RSS_MIB,
        auto=False, run_phase1_fn=lambda pids: calls.append(pids) or "r",
        probe_own_vram_fn=lambda pids: 9999.0,
    )
    assert result == "r"
    assert new_vram == rmp.DEFAULT_PER_WORKER_VRAM_MIB
    assert new_rss == rmp.DEFAULT_PER_WORKER_RSS_MIB
    assert calls == [[]]  # run_phase1_fn still gets a (unused) pids list


def test_run_phase1_serial_path_reports_own_pid(monkeypatch):
    """gpu_workers<=1, no pool -- the caller's own pid is the one holding
    the CUDA context, so pids_out must get it (not stay empty)."""
    monkeypatch.setattr(rmp, "_prewarm_group", lambda *a, **kw: (1, 0))
    pids = []
    rmp.run_phase1([(1, ["AAA"], "AAA", 1.0)], None, "1d", "6m", 100,
                    "/tmp/x", 0, gpu_workers=1, pids_out=pids)
    assert pids == [os.getpid()]


# ============================================================================
# --control-file: live workers/gpu_workers/chunk_size rebalancing
# ============================================================================

def test_control_file_missing_is_silent_and_unchanged(tmp_path):
    w, g, c, changes = rmp._load_control_overrides(
        str(tmp_path / "nope.json"), workers=4, gpu_workers=2, chunk_size=16, cpu_count=16,
    )
    assert (w, g, c, changes) == (4, 2, 16, [])


def test_control_file_applies_valid_values(tmp_path):
    p = tmp_path / "control.json"
    p.write_text(json.dumps({"workers": 3, "gpu_workers": 3, "chunk_size": 8}))
    w, g, c, changes = rmp._load_control_overrides(
        str(p), workers=4, gpu_workers=2, chunk_size=16, cpu_count=16,
    )
    assert (w, g, c) == (3, 3, 8)
    assert changes == ["workers 4 -> 3", "gpu_workers 2 -> 3", "chunk_size 16 -> 8"]


def test_control_file_gpu_workers_auto_string_accepted(tmp_path):
    p = tmp_path / "control.json"
    p.write_text(json.dumps({"gpu_workers": "auto"}))
    w, g, c, changes = rmp._load_control_overrides(
        str(p), workers=4, gpu_workers=2, chunk_size=16, cpu_count=16,
    )
    assert g == "auto"
    assert changes == ["gpu_workers 2 -> auto"]


def test_control_file_unchanged_values_produce_no_change_log(tmp_path):
    p = tmp_path / "control.json"
    p.write_text(json.dumps({"workers": 4}))
    _, _, _, changes = rmp._load_control_overrides(
        str(p), workers=4, gpu_workers=2, chunk_size=16, cpu_count=16,
    )
    assert changes == []


def test_control_file_malformed_json_survives_and_keeps_current(tmp_path, capsys):
    p = tmp_path / "control.json"
    p.write_text("{not valid json")
    w, g, c, changes = rmp._load_control_overrides(
        str(p), workers=4, gpu_workers=2, chunk_size=16, cpu_count=16,
    )
    assert (w, g, c, changes) == (4, 2, 16, [])
    assert "WARNING" in capsys.readouterr().out


def test_control_file_non_object_json_survives_and_keeps_current(tmp_path, capsys):
    p = tmp_path / "control.json"
    p.write_text("[1, 2, 3]")
    w, g, c, changes = rmp._load_control_overrides(
        str(p), workers=4, gpu_workers=2, chunk_size=16, cpu_count=16,
    )
    assert (w, g, c, changes) == (4, 2, 16, [])
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.parametrize("bad_workers", [0, -1, 3.5, "3", None, True, 999])
def test_control_file_wrong_type_or_out_of_range_workers_keeps_current(tmp_path, capsys, bad_workers):
    p = tmp_path / "control.json"
    p.write_text(json.dumps({"workers": bad_workers}))
    w, g, c, changes = rmp._load_control_overrides(
        str(p), workers=4, gpu_workers=2, chunk_size=16, cpu_count=16,
    )
    assert (w, g, c, changes) == (4, 2, 16, [])
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.parametrize("bad_gw", [0, -1, "fast", None, 999])
def test_control_file_wrong_type_or_out_of_range_gpu_workers_keeps_current(tmp_path, capsys, bad_gw):
    p = tmp_path / "control.json"
    p.write_text(json.dumps({"gpu_workers": bad_gw}))
    w, g, c, changes = rmp._load_control_overrides(
        str(p), workers=4, gpu_workers=2, chunk_size=16, cpu_count=16,
    )
    assert (w, g, c, changes) == (4, 2, 16, [])
    assert "WARNING" in capsys.readouterr().out


@pytest.mark.parametrize("bad_cs", [0, -5, 3.5, "16", None])
def test_control_file_wrong_type_or_out_of_range_chunk_size_keeps_current(tmp_path, capsys, bad_cs):
    p = tmp_path / "control.json"
    p.write_text(json.dumps({"chunk_size": bad_cs}))
    w, g, c, changes = rmp._load_control_overrides(
        str(p), workers=4, gpu_workers=2, chunk_size=16, cpu_count=16,
    )
    assert (w, g, c, changes) == (4, 2, 16, [])
    assert "WARNING" in capsys.readouterr().out


def test_control_file_never_raises_on_unreadable_file(tmp_path):
    """A directory where a file is expected -- open() raises IsADirectoryError.
    Must be swallowed like any other malformed-file case, never propagate."""
    p = tmp_path / "control.json"
    p.mkdir()
    w, g, c, changes = rmp._load_control_overrides(
        str(p), workers=4, gpu_workers=2, chunk_size=16, cpu_count=16,
    )
    assert (w, g, c, changes) == (4, 2, 16, [])


def test_warn_if_over_physical_fires_over_budget(capsys):
    rmp._warn_if_over_physical(6, 4, physical=8, context="test:")
    assert "WARNING" in capsys.readouterr().out


def test_warn_if_over_physical_silent_within_budget(capsys):
    rmp._warn_if_over_physical(2, 2, physical=8, context="test:")
    assert capsys.readouterr().out == ""


def test_warn_if_over_physical_skips_auto(capsys):
    """gpu_workers='auto' varies per chunk -- can't sum it, so no warning."""
    rmp._warn_if_over_physical(100, "auto", physical=8, context="test:")
    assert capsys.readouterr().out == ""


# ============================================================================
# estimate_group_vram_mib: per-group VRAM estimate table
# ============================================================================

_BASE = "NeoQuasar/Kronos-base"
_SMALL = "NeoQuasar/Kronos-small"
_MINI = "NeoQuasar/Kronos-mini"


def test_estimate_group_vram_known_models_at_measured_points():
    assert rmp.estimate_group_vram_mib(_BASE, 1) == 1348.0
    assert rmp.estimate_group_vram_mib(_BASE, 4) == 5124.0
    assert rmp.estimate_group_vram_mib(_SMALL, 1) == 682.0
    assert rmp.estimate_group_vram_mib(_SMALL, 4) == 3164.0
    assert rmp.estimate_group_vram_mib(_MINI, 1) == 562.0
    assert rmp.estimate_group_vram_mib(_MINI, 4) == 2818.0


def test_estimate_group_vram_interpolates_n2_n3():
    # base: slope = (5124-1348)/3 = 1258.667/asset
    v2 = rmp.estimate_group_vram_mib(_BASE, 2)
    v3 = rmp.estimate_group_vram_mib(_BASE, 3)
    assert v2 == pytest.approx(1348.0 + 1258.6666666666667)
    assert v3 == pytest.approx(1348.0 + 2 * 1258.6666666666667)
    assert 1348.0 < v2 < v3 < 5124.0


def test_estimate_group_vram_extrapolates_beyond_n4():
    v4 = rmp.estimate_group_vram_mib(_BASE, 4)
    v8 = rmp.estimate_group_vram_mib(_BASE, 8)
    assert v8 > v4


def test_estimate_group_vram_unknown_model_falls_back_to_base():
    unknown = "/local/finetuned/checkpoint-dir"
    assert rmp.estimate_group_vram_mib(unknown, 1) == rmp.estimate_group_vram_mib(_BASE, 1)
    assert rmp.estimate_group_vram_mib(unknown, 4) == rmp.estimate_group_vram_mib(_BASE, 4)


def test_estimate_group_vram_none_model_falls_back_to_base():
    assert rmp.estimate_group_vram_mib(None, 1) == rmp.estimate_group_vram_mib(_BASE, 1)


def test_estimate_group_vram_never_below_n1_value():
    v1 = rmp.estimate_group_vram_mib(_BASE, 1)
    assert rmp.estimate_group_vram_mib(_BASE, 0) >= v1
    assert rmp.estimate_group_vram_mib(_BASE, -5) >= v1


# ============================================================================
# calibrate_vram_table: upward-only refinement of the n=1/n=4 anchors
# ============================================================================

@pytest.fixture(autouse=True)
def _restore_vram_table():
    """calibrate_vram_table() mutates module-level state -- isolate tests."""
    import copy
    saved = copy.deepcopy(rmp._VRAM_TABLE_MIB)
    yield
    rmp._VRAM_TABLE_MIB.clear()
    rmp._VRAM_TABLE_MIB.update(saved)


def test_calibrate_vram_table_raises_known_anchor():
    rmp.calibrate_vram_table(_BASE, 1, 2000.0)
    assert rmp._VRAM_TABLE_MIB[_BASE][1] == 2000.0


def test_calibrate_vram_table_never_lowers():
    rmp.calibrate_vram_table(_BASE, 1, 500.0)  # below the measured 1348.0
    assert rmp._VRAM_TABLE_MIB[_BASE][1] == 1348.0


def test_calibrate_vram_table_ignores_non_anchor_n():
    rmp.calibrate_vram_table(_BASE, 2, 9999.0)
    assert 2 not in rmp._VRAM_TABLE_MIB[_BASE]


def test_calibrate_vram_table_creates_row_for_unknown_model():
    rmp.calibrate_vram_table("/local/checkpoint", 4, 6000.0)
    assert rmp._VRAM_TABLE_MIB["/local/checkpoint"][4] == 6000.0
    # base's own row is untouched
    assert rmp._VRAM_TABLE_MIB[_BASE][4] == 5124.0


# ============================================================================
# pack_waves: first-fit-decreasing bin packing
# ============================================================================

def _g(group_id, n_assets):
    assets = [f"A{group_id}_{i}" for i in range(n_assets)]
    return (group_id, assets, ",".join(assets), 1.0)


def test_pack_waves_largest_first_ordering():
    groups = [_g(1, 1), _g(2, 4), _g(3, 1)]
    # base budget big enough for the 4-asset group (5124) plus headroom, but
    # not both a 4-asset and a 1-asset group together.
    waves = rmp.pack_waves(groups, _BASE, budget_mib=5200, max_concurrent=8)
    assert waves[0][0][0] == 2  # the 4-asset group placed first (largest)


def test_pack_waves_never_exceeds_budget():
    groups = [_g(i, n) for i, n in enumerate([1, 1, 1, 4, 4, 1, 4])]
    budget = 5300.0
    waves = rmp.pack_waves(groups, _BASE, budget_mib=budget, max_concurrent=8)
    for wave in waves:
        total = sum(rmp.estimate_group_vram_mib(_BASE, len(g[1])) for g in wave)
        assert total <= budget + 1e-6


def test_pack_waves_respects_max_concurrent():
    groups = [_g(i, 1) for i in range(10)]  # tiny groups, VRAM never binds
    waves = rmp.pack_waves(groups, _BASE, budget_mib=1_000_000, max_concurrent=3)
    for wave in waves:
        assert len(wave) <= 3


def test_pack_waves_oversized_single_group_gets_own_wave(capsys):
    groups = [_g(1, 4)]  # 5124 MiB, budget too small
    waves = rmp.pack_waves(groups, _BASE, budget_mib=1000.0, max_concurrent=8)
    assert waves == [[groups[0]]]
    assert "WARNING" in capsys.readouterr().out


def test_pack_waves_union_equals_input_no_duplicates_no_losses():
    groups = [_g(i, n) for i, n in enumerate([1, 4, 1, 1, 4, 1, 4, 1])]
    waves = rmp.pack_waves(groups, _BASE, budget_mib=5300.0, max_concurrent=4)
    flattened = [g for wave in waves for g in wave]
    assert sorted(g[0] for g in flattened) == sorted(g[0] for g in groups)
    assert len(flattened) == len(groups)


def test_pack_waves_bimodal_mix_on_5296_budget():
    """10x 1-asset + 10x 4-asset base groups on a 5296 MiB budget (5808 MiB
    card - 512 MiB headroom). 4-asset groups (5124 MiB) can't share a wave
    with anything else at that budget -> 10 waves of size 1. 1-asset groups
    (1348 MiB) pack 3-per-wave (3*1348=4044 <= 5296, 4*1348=5392 > 5296) ->
    4 waves (3,3,3,1). Total: 14 waves for 20 groups -- vs. a single fixed
    scalar sized for the worst case (5124 MiB/worker) giving exactly 1
    worker at a time, 20 waves."""
    groups = [_g(i, 4) for i in range(10)] + [_g(10 + i, 1) for i in range(10)]
    waves = rmp.pack_waves(groups, _BASE, budget_mib=5296.0, max_concurrent=8)
    sizes = sorted(len(w) for w in waves)
    assert sizes == [1] * 10 + [1, 3, 3, 3]
    assert len(waves) == 14
    flattened = [g for wave in waves for g in wave]
    assert sorted(g[0] for g in flattened) == sorted(g[0] for g in groups)


# ============================================================================
# decide_max_concurrent: RAM/physical-core ceiling reused from
# decide_gpu_workers' RAM term (_ram_allows), not a second RAM reader.
# ============================================================================

def test_decide_max_concurrent_ram_bound():
    n = rmp.decide_max_concurrent(avail_ram_mib=5000, per_worker_rss_mib=1100, physical_cores=100)
    assert n == 3  # floor((5000-1024)/1100) = 3, matches decide_gpu_workers' RAM term


def test_decide_max_concurrent_physical_cores_bound():
    n = rmp.decide_max_concurrent(avail_ram_mib=1_000_000, per_worker_rss_mib=100, physical_cores=4)
    assert n == 4


def test_decide_max_concurrent_never_below_one():
    n = rmp.decide_max_concurrent(avail_ram_mib=100, per_worker_rss_mib=1100, physical_cores=8)
    assert n == 1


def test_decide_max_concurrent_matches_decide_gpu_workers_ram_term():
    """Same RAM inputs must produce the same RAM-derived count as
    decide_gpu_workers when VRAM/max_workers aren't binding in either."""
    gw = rmp.decide_gpu_workers(free_vram_mib=1_000_000, avail_ram_mib=6000,
                                 per_worker_vram_mib=1, per_worker_rss_mib=1100, max_workers=1000)
    mc = rmp.decide_max_concurrent(avail_ram_mib=6000, per_worker_rss_mib=1100, physical_cores=1000)
    assert gw == mc
