"""Per-(model, instrument class) strategy stats: classifier, source aggregation,
storage and the read helper.

Phase 1 is production-only: nothing here should change live selection. The
tests that guard that promise live elsewhere and must pass unmodified
(test_disabled_strategy_resolution.py, test_shadow_performance_naive.py).
"""
import os
import sqlite3
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from types import SimpleNamespace

import kairos_pipeline as kp
from kairos_backtest import Direction, asset_class_of_symbol
from kairos_orchestrator import KairosOrchestrator


# --------------------------------------------------------------------------
# 1. Classifier
# --------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,expected", [
    ("BTC-USD", "crypto"), ("MKR-USD", "crypto"), ("SUI20947-USD", "crypto"),
    ("EURUSD=X", "fx_commodity"), ("CADJPY=X", "fx_commodity"),
    ("NG=F", "fx_commodity"), ("CL=F", "fx_commodity"),
    ("GLD", "fx_commodity"), ("UNG", "fx_commodity"), ("CPER", "fx_commodity"),
    ("AAPL", "equity"), ("AAL.L", "equity"), ("0005.HK", "equity"),
    ("BOL.ST", "equity"), ("GMEXICOB.MX", "equity"),
])
def test_classifier(symbol, expected):
    assert asset_class_of_symbol(symbol) == expected


def test_classifier_resolves_symbols_the_membership_classifier_cannot():
    """The pipeline's membership-based classifier returns 'unknown' for symbols
    absent from CANDIDATE_UNIVERSE; this suffix-based one must not."""
    for sym in ("EURUSD=X", "CADJPY=X", "MKR-USD"):
        assert asset_class_of_symbol(sym) != "unknown"


def test_classifier_tolerates_whitespace():
    assert asset_class_of_symbol("  BTC-USD ") == "crypto"


# --------------------------------------------------------------------------
# 2. Per-class aggregation at source
# --------------------------------------------------------------------------

def _bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def _two_bar_df(date0, entry_bar, next_bar):
    return pd.DataFrame(
        [entry_bar, next_bar],
        index=[date0, date0 + pd.Timedelta(days=1)],
    )


def test_signals_are_attributed_to_their_own_symbols_class():
    """A mixed group must split by each signal's symbol, not by a group label."""
    date0 = pd.Timestamp("2026-01-01")
    # LONG, entry ref 100, stop 95, target 110 -> target hit on the next bar
    sigs = [
        (date0, "BTC-USD", "s", Direction.LONG, 95.0, 110.0, 100.0),
        (date0, "AAPL", "s", Direction.LONG, 95.0, 110.0, 100.0),
        (date0, "AAPL", "s", Direction.LONG, 95.0, 110.0, 100.0),
        (date0, "GLD", "s", Direction.LONG, 95.0, 110.0, 100.0),
    ]
    df = _two_bar_df(date0, _bar(99, 101, 98, 100), _bar(100, 120, 99, 118))
    data = {"BTC-USD": df, "AAPL": df, "GLD": df}

    fake = SimpleNamespace(_shadow_signals=sigs, _data_dict=data)
    corpus = KairosOrchestrator._compute_shadow_performance(fake)

    by_class = fake._shadow_performance_by_class
    assert set(by_class["s"]) == {"crypto", "equity", "fx_commodity"}
    assert by_class["s"]["crypto"]["signal_count"] == 1
    assert by_class["s"]["equity"]["signal_count"] == 2
    assert by_class["s"]["fx_commodity"]["signal_count"] == 1
    # the invariant that catches attribution bugs
    assert sum(v["signal_count"] for v in by_class["s"].values()) == corpus["s"]["signal_count"]


def test_naive_evaluator_also_populates_by_class():
    date0 = pd.Timestamp("2026-01-01")
    sigs = [(date0, "BTC-USD", "s", Direction.LONG, 95.0, 110.0, 100.0)]
    df = pd.DataFrame(
        [_bar(99, 101, 98, 100), _bar(102, 106, 101, 104), _bar(106, 130, 104, 128)],
        index=[date0, date0 + pd.Timedelta(days=1), date0 + pd.Timedelta(days=2)],
    )
    fake = SimpleNamespace(_shadow_signals=sigs, _data_dict={"BTC-USD": df})
    corpus = KairosOrchestrator._compute_shadow_performance_naive(fake)

    by_class = fake._shadow_performance_by_class
    assert by_class["s"]["crypto"]["signal_count"] == corpus["s"]["signal_count"]


def test_single_class_group_yields_one_class():
    date0 = pd.Timestamp("2026-01-01")
    sigs = [(date0, "AAPL", "s", Direction.LONG, 95.0, 110.0, 100.0)]
    df = _two_bar_df(date0, _bar(99, 101, 98, 100), _bar(100, 120, 99, 118))
    fake = SimpleNamespace(_shadow_signals=sigs, _data_dict={"AAPL": df})
    KairosOrchestrator._compute_shadow_performance(fake)
    assert list(fake._shadow_performance_by_class["s"]) == ["equity"]


# --------------------------------------------------------------------------
# 3. Export flattening
# --------------------------------------------------------------------------

def test_class_rows_from_export():
    payload = {"shadow_performance_by_class": {
        "s": {"crypto": {"pnl_list": [0.1, -0.05], "sharpe": 1.2, "signal_count": 2}}}}
    rows = kp._class_rows_from_export(payload, ["BTC-USD"], "1d", "6m", "base",
                                      model_path="/some/ckpt")
    assert len(rows) == 1
    r = rows[0]
    assert r["asset_class"] == "crypto"
    assert r["model_path"] == "/some/ckpt"
    assert r["signal_count"] == 2
    assert r["win_rate"] == 0.5
    assert r["stage"] == "base"


def test_legacy_payload_without_the_new_key_yields_no_class_rows():
    """Older exports and hand-built test payloads must not raise."""
    payload = {"summary": {}, "strategy_rankings": [], "shadow_performance": {}}
    assert kp._class_rows_from_export(payload, ["AAPL"], "1d", "6m", "oracle") == []


def test_corpus_rows_unaffected_by_the_new_key():
    payload = {
        "shadow_performance": {"s": {"pnl_list": [0.1], "sharpe": 2.0, "signal_count": 1}},
        "shadow_performance_by_class": {"s": {"equity": {"pnl_list": [0.1], "sharpe": 2.0,
                                                         "signal_count": 1}}},
    }
    rows = kp._rows_from_export(payload, ["AAPL"], "1d", "6m", "oracle")
    assert len(rows) == 1
    assert "asset_class" not in rows[0]


# --------------------------------------------------------------------------
# 4. Storage + read helper
# --------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "t.db")
    c.executescript(kp.SCHEMA)
    return c


def _put(c, run_id, name, cls, sharpe, n, stage="oracle", model_path=None):
    kp.insert_class_stat_row(c, run_id, {
        "stage": stage, "model_path": model_path, "strategy_name": name,
        "asset_class": cls, "sharpe": sharpe, "signal_count": n,
        "win_rate": 0.5, "avg_pnl_per_trade": 0.001,
        "assets": "X", "interval": "1d", "backtest_period": "6m"})


def _put_corpus(c, run_id, name, sharpe, n, stage="oracle"):
    kp.insert_oracle_row(c, run_id, {
        "stage": stage, "strategy_name": name, "sharpe": sharpe, "signal_count": n,
        "win_rate": 0.5, "avg_pnl_per_trade": 0.002,
        "assets": "X", "interval": "1d", "backtest_period": "6m"})


def test_insert_is_idempotent_on_the_natural_key(conn):
    _put(conn, 1, "s", "crypto", 1.0, 10)
    _put(conn, 1, "s", "crypto", 9.9, 10)
    rows = conn.execute("SELECT sharpe FROM strategy_class_stats").fetchall()
    assert rows == [(9.9,)]


def test_read_helper_returns_class_stats_when_thick_enough(conn):
    _put(conn, 1, "s", "crypto", 3.0, 100)
    out = kp.strategy_class_stats(conn, stage="oracle", asset_class="crypto", min_signals=10)
    assert out["s"]["source"] == "class"
    assert out["s"]["sharpe"] == 3.0


def test_read_helper_falls_back_to_corpus_when_thin(conn):
    _put(conn, 1, "s", "crypto", 3.0, 5)
    _put_corpus(conn, 1, "s", -1.0, 500)
    out = kp.strategy_class_stats(conn, stage="oracle", asset_class="crypto", min_signals=30)
    assert out["s"]["source"] == "corpus"
    assert out["s"]["sharpe"] == -1.0


def test_read_helper_can_refuse_to_fall_back(conn):
    _put(conn, 1, "s", "crypto", 3.0, 5)
    _put_corpus(conn, 1, "s", -1.0, 500)
    out = kp.strategy_class_stats(conn, stage="oracle", asset_class="crypto",
                                  min_signals=30, fallback_to_corpus=False)
    assert out == {}


def test_read_helper_corpus_mode_reads_the_corpus_table_not_the_class_table(conn):
    """asset_class=None must never reconstruct corpus from per-class rows --
    Sharpe is a ratio and does not recombine."""
    _put(conn, 1, "s", "crypto", 100.0, 10)
    _put(conn, 1, "s", "equity", 100.0, 10)
    _put_corpus(conn, 1, "s", 0.5, 20)
    out = kp.strategy_class_stats(conn, stage="oracle", asset_class=None)
    assert out["s"]["sharpe"] == 0.5
    assert out["s"]["source"] == "corpus"


def test_read_helper_separates_models(conn):
    _put(conn, 1, "s", "crypto", 1.0, 100, stage="base", model_path=None)
    _put(conn, 2, "s", "crypto", 7.0, 100, stage="base", model_path="/ckpt")
    ft = kp.strategy_class_stats(conn, stage="base", asset_class="crypto",
                                 model_path="/ckpt", min_signals=10)
    assert ft["s"]["sharpe"] == 7.0


def test_read_helper_weights_by_signal_count(conn):
    _put(conn, 1, "s", "crypto", 0.0, 100)
    _put(conn, 2, "s", "crypto", 10.0, 300)
    out = kp.strategy_class_stats(conn, stage="oracle", asset_class="crypto", min_signals=10)
    assert out["s"]["sharpe"] == pytest.approx(7.5)
    assert out["s"]["signal_count"] == 400
    assert out["s"]["n_groups"] == 2
