"""Tests for scripts/run_model_parallel.py.

No GPU/model download needed: enumerate_backtest_dates() is pure pandas date
math, and the orchestrator-parity test drives the real KairosOrchestrator
with a stub batch_predict_fn that never touches a real model (config.
no_prediction stays False so _run_day routes through
multi_predictor.predict_all(), but the stub returns {} immediately, so no
strategy ever evaluates). select_prioritized_groups()/_chunked() are pure
DB/list logic.
"""
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
