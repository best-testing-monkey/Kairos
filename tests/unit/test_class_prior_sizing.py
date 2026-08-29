"""Item 5: allocation shrinks toward the (model, class) base rate, not a flat 0.5.

The safety property under test throughout: with no class prior available, every
number must be arithmetically identical to the previous behaviour.
"""
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import kairos_pipeline as kp
import kairos_signals
from allocation import AllocationConfig, Candidate, compute_derived


def _cand(**kw):
    base = dict(
        ticker="BTC-USD", strategy="s", direction="LONG",
        entry=100.0, stop=95.0, target=110.0,
        ev_pct=1.0, n=10, base_win_rate=0.8,
        backtest_period="6m", sharpe=1.0, advised_liquidity_pct=100.0,
    )
    base.update(kw)
    return Candidate(**base)


CFG = AllocationConfig()


# --------------------------------------------------------------------------
# compute_derived
# --------------------------------------------------------------------------

def test_no_class_prior_is_identical_to_shrinking_toward_half():
    c = _cand(class_prior_win_rate=None)
    d = compute_derived(c, CFG)
    shrink = c.n / (c.n + CFG.n0)
    assert d["p_shrunk"] == pytest.approx(0.5 + (c.base_win_rate - 0.5) * shrink)


def test_class_prior_moves_the_shrink_target():
    c = _cand(class_prior_win_rate=0.35)
    d = compute_derived(c, CFG)
    shrink = c.n / (c.n + CFG.n0)
    assert d["p_shrunk"] == pytest.approx(0.35 + (c.base_win_rate - 0.35) * shrink)


def test_a_thin_strategy_regresses_toward_its_class_not_a_coin_flip():
    """n=1 -> almost all prior. A bad class must drag it below the 0.5 default."""
    weak = compute_derived(_cand(n=1, class_prior_win_rate=0.30), CFG)
    default = compute_derived(_cand(n=1, class_prior_win_rate=None), CFG)
    assert weak["p_shrunk"] < default["p_shrunk"]
    assert weak["kelly_frac"] <= default["kelly_frac"]


def test_a_thick_strategy_is_barely_moved_by_the_prior():
    """Evidence dominates: n >> n0 means the prior is nearly irrelevant."""
    a = compute_derived(_cand(n=100000, class_prior_win_rate=0.30), CFG)
    b = compute_derived(_cand(n=100000, class_prior_win_rate=None), CFG)
    assert abs(a["p_shrunk"] - b["p_shrunk"]) < 1e-3


def test_prior_equal_to_half_is_a_no_op():
    a = compute_derived(_cand(class_prior_win_rate=0.5), CFG)
    b = compute_derived(_cand(class_prior_win_rate=None), CFG)
    assert a["p_shrunk"] == pytest.approx(b["p_shrunk"])


# --------------------------------------------------------------------------
# _class_prior_win_rate
# --------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    c = sqlite3.connect(p)
    c.executescript(kp.SCHEMA)
    c.close()
    return str(p)


def _seed(db, rows):
    c = sqlite3.connect(db)
    for i, r in enumerate(rows):
        kp.insert_class_stat_row(c, i, {
            "stage": r.get("stage", "base"), "model_path": r.get("model_path"),
            "strategy_name": r.get("strategy_name", f"s{i}"),
            "asset_class": r["asset_class"], "sharpe": 1.0,
            "signal_count": r["signal_count"], "win_rate": r["win_rate"],
            "avg_pnl_per_trade": 0.001, "assets": "X",
            "interval": r.get("interval", "1d"), "backtest_period": "6m"})
    c.commit(); c.close()


def test_prior_is_signal_count_weighted(db):
    _seed(db, [
        {"asset_class": "crypto", "signal_count": 100, "win_rate": 0.40},
        {"asset_class": "crypto", "signal_count": 300, "win_rate": 0.60},
    ])
    got = kairos_signals._class_prior_win_rate("1d", ["BTC-USD"], None, db_path=db)
    assert got == pytest.approx((0.40 * 100 + 0.60 * 300) / 400)


def test_mixed_class_group_has_no_prior(db):
    _seed(db, [{"asset_class": "crypto", "signal_count": 100, "win_rate": 0.6}])
    assert kairos_signals._class_prior_win_rate(
        "1d", ["BTC-USD", "AAPL"], None, db_path=db) is None


def test_other_classes_do_not_leak_in(db):
    _seed(db, [
        {"asset_class": "crypto", "signal_count": 100, "win_rate": 0.60},
        {"asset_class": "equity", "signal_count": 900, "win_rate": 0.10},
    ])
    assert kairos_signals._class_prior_win_rate(
        "1d", ["BTC-USD"], None, db_path=db) == pytest.approx(0.60)


def test_model_is_distinguished(db):
    _seed(db, [
        {"asset_class": "crypto", "signal_count": 100, "win_rate": 0.20},
        {"asset_class": "crypto", "signal_count": 100, "win_rate": 0.90,
         "stage": "finetuned", "model_path": "/ckpt"},
    ])
    assert kairos_signals._class_prior_win_rate(
        "1d", ["BTC-USD"], "/ckpt", db_path=db) == pytest.approx(0.90)
    assert kairos_signals._class_prior_win_rate(
        "1d", ["BTC-USD"], None, db_path=db) == pytest.approx(0.20)


def test_empty_and_broken_db_return_none(db, tmp_path):
    assert kairos_signals._class_prior_win_rate("1d", ["BTC-USD"], None, db_path=db) is None
    bad = tmp_path / "nope.db"
    bad.write_text("not a database")
    assert kairos_signals._class_prior_win_rate(
        "1d", ["BTC-USD"], None, db_path=str(bad)) is None
