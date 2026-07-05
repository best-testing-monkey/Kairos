import numpy as np
import pandas as pd
import pytest

import kairos_pipeline as kp


# ============================================================================
# Liquidity pass/fail logic
# ============================================================================

def test_liquidity_equity_pass_above_threshold():
    passed, reason, note = kp.evaluate_liquidity(
        "SPY", "equity", bars=250, dollar_volume=60_000_000, ann_vol=0.2, atr_pct=1.0
    )
    assert passed is True
    assert reason is None


def test_liquidity_equity_fail_below_threshold():
    passed, reason, note = kp.evaluate_liquidity(
        "XYZ", "equity", bars=250, dollar_volume=10_000_000, ann_vol=0.2, atr_pct=1.0
    )
    assert passed is False
    assert "low_dollar_volume" in reason


def test_liquidity_crypto_pass_above_threshold():
    passed, reason, note = kp.evaluate_liquidity(
        "BTC-USD", "crypto", bars=250, dollar_volume=15_000_000, ann_vol=0.5, atr_pct=2.0
    )
    assert passed is True


def test_liquidity_crypto_fail_below_threshold():
    passed, reason, note = kp.evaluate_liquidity(
        "SHIB-USD", "crypto", bars=250, dollar_volume=1_000_000, ann_vol=0.5, atr_pct=2.0
    )
    assert passed is False
    assert "low_dollar_volume" in reason


def test_liquidity_fx_exempt_from_dollar_volume():
    passed, reason, note = kp.evaluate_liquidity(
        "EURUSD=X", "fx_commodity", bars=250, dollar_volume=0, ann_vol=0.05, atr_pct=0.6
    )
    assert passed is True
    assert note == "fx_exempt_from_dollar_volume_filter"


def test_liquidity_fail_low_atr():
    passed, reason, note = kp.evaluate_liquidity(
        "SPY", "equity", bars=250, dollar_volume=100_000_000, ann_vol=0.1, atr_pct=0.1
    )
    assert passed is False
    assert "low_atr_pct" in reason


def test_liquidity_fail_insufficient_bars():
    passed, reason, note = kp.evaluate_liquidity(
        "SPY", "equity", bars=50, dollar_volume=100_000_000, ann_vol=0.1, atr_pct=1.0
    )
    assert passed is False
    assert "insufficient_bars" in reason


# ============================================================================
# Correlation-pair math
# ============================================================================

def test_correlation_perfectly_correlated_series():
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    base = pd.Series(np.linspace(100, 200, 200), index=dates)
    a = base
    b = base * 2.0  # perfectly correlated (proportional) returns
    full_corr, roll_median, overlap = kp.compute_pair_correlation(a, b)
    assert overlap == 200
    assert full_corr == pytest.approx(1.0, abs=1e-6)


def test_correlation_uncorrelated_series():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    a = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)), index=dates)
    b = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)), index=dates)
    full_corr, roll_median, overlap = kp.compute_pair_correlation(a, b)
    assert overlap == 300
    assert abs(full_corr) < 0.5


def test_correlation_insufficient_overlap_returns_none():
    dates_a = pd.date_range("2024-01-01", periods=100, freq="D")
    dates_b = pd.date_range("2024-06-01", periods=100, freq="D")
    a = pd.Series(np.linspace(100, 200, 100), index=dates_a)
    b = pd.Series(np.linspace(50, 60, 100), index=dates_b)
    full_corr, roll_median, overlap = kp.compute_pair_correlation(a, b, min_overlap=150)
    assert full_corr is None
    assert roll_median is None


# ============================================================================
# Greedy grouping logic
# ============================================================================

def test_greedy_grouping_forms_expected_cluster():
    pairs = [
        {"symbol_a": "A", "symbol_b": "B", "asset_class": "crypto", "full_corr": 0.9},
        {"symbol_a": "B", "symbol_b": "C", "asset_class": "crypto", "full_corr": 0.8},
        {"symbol_a": "A", "symbol_b": "C", "asset_class": "crypto", "full_corr": 0.75},
        {"symbol_a": "X", "symbol_b": "Y", "asset_class": "crypto", "full_corr": 0.3},  # below threshold
    ]
    groups = kp.greedy_group_pairs(pairs, min_abs_corr=0.6, max_group_size=4)
    assert len(groups) == 1
    g = groups[0]
    assert set(g["symbols"]) == {"A", "B", "C"}
    assert g["asset_class"] == "crypto"
    assert g["mean_intra_corr"] > 0.6


def test_greedy_grouping_respects_max_group_size():
    # 5 mutually strongly-correlated symbols; group should cap at 4 members.
    syms = ["A", "B", "C", "D", "E"]
    pairs = []
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            pairs.append({
                "symbol_a": syms[i], "symbol_b": syms[j],
                "asset_class": "equity", "full_corr": 0.95 - 0.01 * (i + j),
            })
    groups = kp.greedy_group_pairs(pairs, min_abs_corr=0.6, max_group_size=4)
    assert all(len(g["symbols"]) <= 4 for g in groups)


def test_greedy_grouping_no_pairs_above_threshold():
    pairs = [
        {"symbol_a": "A", "symbol_b": "B", "asset_class": "crypto", "full_corr": 0.1},
    ]
    groups = kp.greedy_group_pairs(pairs, min_abs_corr=0.6)
    assert groups == []


# ============================================================================
# DB schema round-trip
# ============================================================================

def test_db_schema_round_trip(tmp_path):
    db_path = str(tmp_path / "pipeline_test.db")
    conn = kp.get_connection(db_path)
    run_id = kp.start_run(conn, "universe", "1d", {"foo": "bar"})
    assert isinstance(run_id, int)

    kp.insert_universe_row(conn, run_id, {
        "symbol": "SPY", "asset_class": "equity", "bars": 250,
        "dollar_volume": 1e8, "ann_vol": 0.2, "atr_pct": 1.1,
        "interval_probe_ok": True, "liquidity_note": None,
        "passed": True, "fail_reason": None,
    })
    kp.insert_correlation_row(conn, run_id, {
        "symbol_a": "SPY", "symbol_b": "QQQ", "asset_class": "equity",
        "full_corr": 0.87, "rolling_corr_median": 0.9, "overlap_bars": 200,
    })
    kp.insert_group_row(conn, run_id, {
        "group_id": 1, "asset_class": "equity", "symbols": "SPY,QQQ", "mean_intra_corr": 0.87,
    })
    kp.insert_oracle_row(conn, run_id, {
        "stage": "oracle", "strategy_name": "trend_follow", "sharpe": 1.23,
        "signal_count": 10, "win_rate": 0.6, "avg_pnl_per_trade": 0.01,
        "assets": "SPY,QQQ", "interval": "1d", "backtest_period": "3m",
    })
    kp.insert_model_row(conn, run_id, {
        "stage": "base", "strategy_name": "trend_follow", "sharpe": 0.9,
        "signal_count": 8, "win_rate": 0.5, "avg_pnl_per_trade": 0.005,
        "assets": "SPY,QQQ", "interval": "1d", "backtest_period": "3m",
        "model_path": None,
    })
    conn.commit()

    assert conn.execute("SELECT symbol FROM universe_screen WHERE run_id=?", (run_id,)).fetchone()[0] == "SPY"
    row = conn.execute(
        "SELECT symbol_a, symbol_b, full_corr FROM correlation_pairs WHERE run_id=?", (run_id,)
    ).fetchone()
    assert row == ("SPY", "QQQ", 0.87)
    row = conn.execute(
        "SELECT group_id, symbols, mean_intra_corr FROM suggested_groups WHERE run_id=?", (run_id,)
    ).fetchone()
    assert row == (1, "SPY,QQQ", 0.87)
    row = conn.execute(
        "SELECT strategy_name, sharpe, signal_count FROM oracle_results WHERE run_id=?", (run_id,)
    ).fetchone()
    assert row == ("trend_follow", 1.23, 10)
    row = conn.execute(
        "SELECT stage, strategy_name, sharpe, model_path FROM model_results WHERE run_id=?", (run_id,)
    ).fetchone()
    assert row == ("base", "trend_follow", 0.9, None)

    conn.close()
