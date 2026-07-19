#!/usr/bin/env python3
"""
kairos_pipeline.py - Staged asset-discovery pipeline for Kairos.

Stages:
  1. universe     - screen a curated candidate universe for liquidity/volatility.
  2. correlation  - compute pairwise correlations among stage-1 survivors and
                     greedily cluster them into suggested trading groups.
  3. oracle       - run strategy/kairos_strategies.py with --no-prediction
                     (oracle baseline, no GPU/model needed) as a subprocess and
                     ingest the exported JSON results.
  4. base         - same subprocess runner, WITHOUT --no-prediction, using the
                     default (base) Kronos model. Requires GPU - not executed
                     in this environment, but fully wired.
  5. finetuned    - same as `base` but passes --model <path> to use a
                     finetuned Kronos checkpoint. Requires GPU - not executed
                     in this environment, but fully wired.

Results are persisted to a SQLite DB (data/pipeline_results.db) and mirrored
to per-stage CSV files under results/.

Usage:
    uv run ./strategy/kairos_pipeline.py --stage universe
    uv run ./strategy/kairos_pipeline.py --stage correlation
    uv run ./strategy/kairos_pipeline.py --stage oracle --assets BTC-USD ETH-USD SOL-USD
    uv run ./strategy/kairos_pipeline.py --stage oracle --group_id 3
"""
import sys
import os

# strategy/ has no __init__.py - it is not a package. Scripts and tests add it
# to sys.path explicitly (see tests/conftest.py and kairos_strategies.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append("..")

import argparse
import csv
import fcntl
import json
import sqlite3
import subprocess
import sys as _sys
import tempfile
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

import price_cache
from kairos_strategies import asset_class_for, _period_to_weeks, _parse_period

# ── Paths ──────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "data", "pipeline_results.db")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
STRATEGIES_SCRIPT = os.path.join(REPO_ROOT, "strategy", "kairos_strategies.py")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
FINETUNE_LOCK_PATH = os.path.join(REPO_ROOT, "data", "finetune_next.lock")

# ── Candidate universe ───────────────────────────────────────────────────────

CANDIDATE_UNIVERSE = {
    "crypto": [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD",
        "AVAX-USD", "LINK-USD", "DOT-USD", "POL28321-USD", "LTC-USD", "BCH-USD",
        "ATOM-USD", "UNI7083-USD", "NEAR-USD", "ARB-USD", "OP-USD",
        "INJ-USD", "SUI20947-USD", "FIL-USD", "ICP-USD", "ETC-USD", "XLM-USD",
        "HBAR-USD", "VET-USD", "ALGO-USD", "AAVE-USD", "MKR-USD", "GRT6719-USD",
        "SAND-USD", "MANA-USD", "AXS-USD", "EOS-USD", "XTZ-USD", "THETA-USD",
        "RUNE-USD", "CRV-USD", "SNX-USD", "ENJ-USD",
        "CHZ-USD", "ZEC-USD", "DASH-USD", "KAVA-USD", "1INCH-USD",
        "LDO-USD", "PEPE24478-USD",
        "BEAM-USD", "BNB-USD", "BONK-USD", "GALA-USD", "JTO-USD", "JUP-USD",
        "ONDO-USD", "PYTH-USD", "RENDER-USD", "SHIB-USD", "STRK-USD", "TIA-USD",
        "TON-USD", "USUAL-USD", "WIF-USD", "WLD-USD",
    ],
    "equity": [
        "SPY", "QQQ", "IWM", "DIA",
        "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM", "XOM",
        "TLT", "HYG",
        "ABBV", "ABT", "AMD", "BA", "BLK", "BX", "CAT", "CB", "COST", "CRM",
        "DE", "GS", "HD", "JNJ", "KO", "LLY", "MA", "MCD", "MMM", "MRK",
        "MS", "NFLX", "ORCL", "PG", "RTX", "SYK", "TMO", "UNH", "V", "VRTX",
    ],
    "fx_commodity": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
        "USDCHF=X", "NZDUSD=X",
        "EURJPY=X", "EURGBP=X", "GBPJPY=X", "AUDCAD=X", "AUDNZD=X",
        "CADJPY=X", "NZDJPY=X", "CHFJPY=X", "EURCAD=X",
        "GLD", "SLV", "USO", "UNG", "DBC", "GDX",
        "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZW=F", "ZS=F",
        "PDBC", "CPER", "COPX", "REMX",
    ],
}

FX_SUFFIX = "=X"

# ── DB schema ────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT,
    timestamp TEXT,
    interval TEXT,
    params_json TEXT
);

CREATE TABLE IF NOT EXISTS universe_screen (
    run_id INTEGER,
    symbol TEXT,
    asset_class TEXT,
    bars INTEGER,
    dollar_volume REAL,
    ann_vol REAL,
    atr_pct REAL,
    interval_probe_ok INTEGER,
    liquidity_note TEXT,
    passed INTEGER,
    fail_reason TEXT
);

CREATE TABLE IF NOT EXISTS correlation_pairs (
    run_id INTEGER,
    symbol_a TEXT,
    symbol_b TEXT,
    asset_class TEXT,
    full_corr REAL,
    rolling_corr_median REAL,
    overlap_bars INTEGER
);

CREATE TABLE IF NOT EXISTS suggested_groups (
    run_id INTEGER,
    group_id INTEGER,
    asset_class TEXT,
    symbols TEXT,
    mean_intra_corr REAL
);

CREATE TABLE IF NOT EXISTS oracle_results (
    run_id INTEGER,
    stage TEXT,
    strategy_name TEXT,
    sharpe REAL,
    signal_count INTEGER,
    win_rate REAL,
    avg_pnl_per_trade REAL,
    assets TEXT,
    interval TEXT,
    backtest_period TEXT
);

CREATE TABLE IF NOT EXISTS model_results (
    run_id INTEGER,
    stage TEXT,
    strategy_name TEXT,
    sharpe REAL,
    signal_count INTEGER,
    win_rate REAL,
    avg_pnl_per_trade REAL,
    assets TEXT,
    interval TEXT,
    backtest_period TEXT,
    model_path TEXT
);

CREATE TABLE IF NOT EXISTS disabled_strategies (
    interval TEXT NOT NULL,
    assets TEXT NOT NULL,           -- sorted CSV, normalized
    strategy_name TEXT NOT NULL,
    avg_pnl_per_trade REAL,
    sharpe REAL,
    signal_count INTEGER,
    source_run_id INTEGER,
    updated_at TEXT,
    PRIMARY KEY (interval, assets, strategy_name)
);

CREATE TABLE IF NOT EXISTS viability_report (
    run_id INTEGER,
    strategy_name TEXT,
    assets TEXT,
    asset_class TEXT,
    interval TEXT,
    backtest_period TEXT,
    oracle_sharpe REAL,
    oracle_signals INTEGER,
    oracle_win_rate REAL,
    oracle_avg_pnl_per_trade REAL,
    oracle_run_id INTEGER,
    base_sharpe REAL,
    base_signals INTEGER,
    base_win_rate REAL,
    base_avg_pnl_per_trade REAL,
    base_run_id INTEGER,
    base_model_path TEXT,
    signals_per_week REAL,
    viable INTEGER
);

CREATE TABLE IF NOT EXISTS finetuned_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assets TEXT NOT NULL,            -- sorted CSV (canonical registry key)
    assets_raw TEXT NOT NULL,        -- as used in oracle/model_results rows
    interval TEXT NOT NULL,
    backtest_period TEXT NOT NULL,
    train_start TEXT, train_end TEXT,
    test_start TEXT, test_end TEXT,
    model_path TEXT,
    status TEXT NOT NULL,            -- training | accepted | rejected | failed
    base_run_id INTEGER, finetuned_run_id INTEGER,
    base_viable_count INTEGER, ft_viable_count INTEGER,
    base_mean_sharpe REAL, ft_mean_sharpe REAL,
    created_at TEXT,
    UNIQUE(assets, interval)
);
"""


def get_connection(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def start_run(conn, stage, interval, params: dict) -> int:
    cur = conn.execute(
        "INSERT INTO runs (stage, timestamp, interval, params_json) VALUES (?, ?, ?, ?)",
        (stage, datetime.now().isoformat(timespec="seconds"), interval, json.dumps(params)),
    )
    conn.commit()
    return cur.lastrowid


# ── Insert helpers (also used by tests for DB round-trip checks) ────────────

def insert_universe_row(conn, run_id, row: dict):
    conn.execute(
        """INSERT INTO universe_screen
           (run_id, symbol, asset_class, bars, dollar_volume, ann_vol, atr_pct,
            interval_probe_ok, liquidity_note, passed, fail_reason)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, row["symbol"], row["asset_class"], row.get("bars"),
         row.get("dollar_volume"), row.get("ann_vol"), row.get("atr_pct"),
         int(bool(row.get("interval_probe_ok"))), row.get("liquidity_note"),
         int(bool(row.get("passed"))), row.get("fail_reason")),
    )


def insert_correlation_row(conn, run_id, row: dict):
    conn.execute(
        """INSERT INTO correlation_pairs
           (run_id, symbol_a, symbol_b, asset_class, full_corr, rolling_corr_median, overlap_bars)
           VALUES (?,?,?,?,?,?,?)""",
        (run_id, row["symbol_a"], row["symbol_b"], row["asset_class"],
         row.get("full_corr"), row.get("rolling_corr_median"), row.get("overlap_bars")),
    )


def insert_group_row(conn, run_id, row: dict):
    conn.execute(
        """INSERT INTO suggested_groups (run_id, group_id, asset_class, symbols, mean_intra_corr)
           VALUES (?,?,?,?,?)""",
        (run_id, row["group_id"], row["asset_class"], row["symbols"], row.get("mean_intra_corr")),
    )


def insert_oracle_row(conn, run_id, row: dict):
    conn.execute(
        """INSERT INTO oracle_results
           (run_id, stage, strategy_name, sharpe, signal_count, win_rate, avg_pnl_per_trade,
            assets, interval, backtest_period)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (run_id, row.get("stage", "oracle"), row["strategy_name"], row.get("sharpe"),
         row.get("signal_count"), row.get("win_rate"), row.get("avg_pnl_per_trade"),
         row.get("assets"), row.get("interval"), row.get("backtest_period")),
    )


def insert_model_row(conn, run_id, row: dict):
    conn.execute(
        """INSERT INTO model_results
           (run_id, stage, strategy_name, sharpe, signal_count, win_rate, avg_pnl_per_trade,
            assets, interval, backtest_period, model_path)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, row["stage"], row["strategy_name"], row.get("sharpe"),
         row.get("signal_count"), row.get("win_rate"), row.get("avg_pnl_per_trade"),
         row.get("assets"), row.get("interval"), row.get("backtest_period"), row.get("model_path")),
    )


def insert_finetune_registry_row(conn, row: dict) -> int:
    """Insert a new finetuned_models row and return its id.

    `row` must supply assets (sorted CSV key), assets_raw, interval,
    backtest_period, and status; all other columns are optional. `created_at`
    defaults to now if not supplied.
    """
    cur = conn.execute(
        """INSERT INTO finetuned_models
           (assets, assets_raw, interval, backtest_period, train_start, train_end,
            test_start, test_end, model_path, status, base_run_id, finetuned_run_id,
            base_viable_count, ft_viable_count, base_mean_sharpe, ft_mean_sharpe, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (row["assets"], row["assets_raw"], row["interval"], row["backtest_period"],
         row.get("train_start"), row.get("train_end"), row.get("test_start"), row.get("test_end"),
         row.get("model_path"), row["status"], row.get("base_run_id"), row.get("finetuned_run_id"),
         row.get("base_viable_count"), row.get("ft_viable_count"),
         row.get("base_mean_sharpe"), row.get("ft_mean_sharpe"),
         row.get("created_at") or datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    return cur.lastrowid


def update_finetune_registry_row(conn, row_id: int, **fields) -> None:
    """Update arbitrary columns of a finetuned_models row by id."""
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE finetuned_models SET {set_clause} WHERE id = ?",
        (*fields.values(), row_id),
    )
    conn.commit()


def refresh_disabled_strategies(conn, run_id, rows, interval, assets, min_signals=5):
    """
    Replace the disabled_strategies rows for one (interval, assets) profile
    with the current criterion applied to `rows` (the same oracle_results row
    dicts that were just inserted for this run).

    A strategy is disabled for this profile if its `avg_pnl_per_trade` is
    negative AND its `signal_count` is >= `min_signals` (avoids disabling on
    noise from a handful of trades). This is a full replace, not a merge: any
    strategy previously disabled for this profile that no longer meets the
    criterion (positive avg_pnl_per_trade, too few signals, or simply absent
    from `rows`) is dropped from the table, i.e. re-enabled.

    `assets` is normalized to a sorted CSV key internally - callers do not
    need to pre-sort it.

    Returns (newly_disabled, re_enabled): two sorted lists of strategy names -
    newly_disabled = names in the new set but not the old set, re_enabled =
    names in the old set but not the new set.
    """
    assets_key = ",".join(sorted(assets))

    old_set = {
        r[0] for r in conn.execute(
            "SELECT strategy_name FROM disabled_strategies WHERE interval=? AND assets=?",
            (interval, assets_key),
        ).fetchall()
    }

    conn.execute(
        "DELETE FROM disabled_strategies WHERE interval=? AND assets=?",
        (interval, assets_key),
    )

    updated_at = datetime.now().isoformat(timespec="seconds")
    new_set = set()
    for row in rows:
        avg_pnl = row.get("avg_pnl_per_trade")
        signal_count = row.get("signal_count") or 0
        if avg_pnl is not None and avg_pnl < 0 and signal_count >= min_signals:
            name = row["strategy_name"]
            new_set.add(name)
            conn.execute(
                """INSERT INTO disabled_strategies
                   (interval, assets, strategy_name, avg_pnl_per_trade, sharpe,
                    signal_count, source_run_id, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (interval, assets_key, name, avg_pnl, row.get("sharpe"),
                 signal_count, run_id, updated_at),
            )

    conn.commit()

    newly_disabled = sorted(new_set - old_set)
    re_enabled = sorted(old_set - new_set)
    return newly_disabled, re_enabled


def dump_csv(table: str, rows: list, stage: str):
    """Write newly-inserted rows for a table to results/<stage>_<table>_<timestamp>.csv.

    The table name is included because a single stage (e.g. correlation) can
    write to more than one table (pairs + groups) within the same second,
    which would otherwise collide on the timestamp-only filename.
    """
    if not rows:
        return None
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = os.path.join(
        RESULTS_DIR, f"{stage}_{table}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    # Rows may have heterogeneous keys (e.g. universe-screen rows for symbols
    # with no data omit bars/atr_pct/dollar_volume/ann_vol/liquidity_note) —
    # use the union of keys across all rows, in first-seen order, so
    # DictWriter never chokes on a later row with extra fields.
    fieldnames = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return fname


# ── Stage 1: universe screen ─────────────────────────────────────────────────

def asset_class_of(symbol: str) -> str:
    for cls, syms in CANDIDATE_UNIVERSE.items():
        if symbol in syms:
            return cls
    return "unknown"


def is_fx(symbol: str, asset_class: str) -> bool:
    return asset_class == "fx_commodity" and symbol.endswith(FX_SUFFIX)


def liquidity_threshold(asset_class: str) -> float:
    """Minimum median daily dollar volume, by asset class. FX is exempt (0)."""
    if asset_class == "crypto":
        return 10_000_000.0
    if asset_class == "equity":
        return 50_000_000.0
    return 0.0  # fx / commodities handled separately (ETFs like GLD still use equity-style vol)


def evaluate_liquidity(symbol: str, asset_class: str, bars: int, dollar_volume,
                        ann_vol, atr_pct, min_bars: int = 200, atr_min: float = 0.5):
    """
    Pure logic (no I/O) so it can be unit-tested with synthetic inputs.

    Returns (passed: bool, fail_reason: str or None, liquidity_note: str or None).
    """
    is_fx_symbol = is_fx(symbol, asset_class) or symbol.endswith(FX_SUFFIX)

    if bars < min_bars:
        return False, f"insufficient_bars({bars}<{min_bars})", None

    liquidity_note = None
    if is_fx_symbol:
        # FX pairs report zero/NaN volume from yfinance; exempt from the
        # dollar-volume filter and record a note instead.
        liquidity_note = "fx_exempt_from_dollar_volume_filter"
    else:
        threshold = liquidity_threshold(asset_class)
        if dollar_volume is None or dollar_volume < threshold:
            return False, f"low_dollar_volume({dollar_volume}<{threshold})", liquidity_note

    if atr_pct is None or atr_pct < atr_min:
        return False, f"low_atr_pct({atr_pct}<{atr_min})", liquidity_note

    return True, None, liquidity_note


def compute_universe_stats(df: pd.DataFrame):
    """Compute bars, dollar_volume, ann_vol, atr_pct from a raw OHLCV frame."""
    bars = len(df)
    close = df["close"].astype(float)
    if "volume" in df.columns:
        dollar_volume = float((close * df["volume"].astype(float)).median())
    else:
        dollar_volume = None

    log_ret = np.log(close / close.shift(1)).dropna()
    ann_vol = float(log_ret.std() * np.sqrt(252)) if len(log_ret) > 1 else None

    if {"high", "low"}.issubset(df.columns):
        high, low, prev_close = df["high"].astype(float), df["low"].astype(float), close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = float(atr / close.iloc[-1] * 100.0) if pd.notna(atr) and close.iloc[-1] else None
    else:
        atr_pct = None

    return bars, dollar_volume, ann_vol, atr_pct


def run_stage_universe(conn, interval="1d"):
    price_cache.configure(remote=False)
    run_id = start_run(conn, "universe", interval, {"interval": interval})
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=400)  # ~1y + buffer

    inserted_rows = []
    for asset_class, symbols in CANDIDATE_UNIVERSE.items():
        for symbol in symbols:
            row = {"symbol": symbol, "asset_class": asset_class, "passed": False}
            try:
                df = price_cache.get_price_data(
                    symbol, start_date=start_dt.isoformat(), end_date=end_dt.isoformat(),
                    interval="1d",
                )
                if df is None or df.empty:
                    row["fail_reason"] = "no_data_returned"
                    row["interval_probe_ok"] = False
                else:
                    df = df.sort_index().copy()
                    df.columns = [c.lower() for c in df.columns]
                    idx = pd.to_datetime(df.index)
                    df.index = idx.tz_convert(None) if idx.tz is not None else idx

                    bars, dollar_volume, ann_vol, atr_pct = compute_universe_stats(df)
                    row.update(bars=bars, dollar_volume=dollar_volume, ann_vol=ann_vol, atr_pct=atr_pct)

                    # Probe the requested --interval separately (may differ from 1d).
                    interval_probe_ok = True
                    if interval != "1d":
                        try:
                            probe_start = end_dt - timedelta(days=5)
                            probe = price_cache.get_price_data(
                                symbol, start_date=probe_start.isoformat(),
                                end_date=end_dt.isoformat(), interval=interval,
                            )
                            interval_probe_ok = probe is not None and not probe.empty
                        except Exception:
                            interval_probe_ok = False
                    row["interval_probe_ok"] = interval_probe_ok

                    passed, fail_reason, liquidity_note = evaluate_liquidity(
                        symbol, asset_class, bars, dollar_volume, ann_vol, atr_pct
                    )
                    row["passed"] = passed
                    row["fail_reason"] = fail_reason
                    row["liquidity_note"] = liquidity_note
            except Exception as exc:
                row["fail_reason"] = f"fetch_error: {exc}"
                row["interval_probe_ok"] = False

            insert_universe_row(conn, run_id, row)
            inserted_rows.append({"run_id": run_id, **row})
            status = "PASS" if row.get("passed") else "fail"
            print(f"  [{asset_class:>13}] {symbol:<10} {status:5} "
                  f"bars={row.get('bars')} $vol={row.get('dollar_volume')} "
                  f"atr%={row.get('atr_pct')} reason={row.get('fail_reason')}")

    conn.commit()
    csv_path = dump_csv("universe_screen", inserted_rows, "universe")
    n_pass = sum(1 for r in inserted_rows if r.get("passed"))
    print(f"\nStage 1 (universe) done: {n_pass}/{len(inserted_rows)} passed. "
          f"run_id={run_id}. CSV: {csv_path}")
    return run_id


# ── Stage 2: correlation screen ──────────────────────────────────────────────

def compute_pair_correlation(series_a: pd.Series, series_b: pd.Series, min_overlap=150, roll_window=30):
    """
    Pure logic on two aligned close-price series (indexed by date).
    Returns (full_corr, rolling_corr_median, overlap_bars) or (None, None, n) if
    insufficient overlap.
    """
    aligned = pd.concat([series_a, series_b], axis=1, join="inner").dropna()
    overlap_bars = len(aligned)
    if overlap_bars < min_overlap:
        return None, None, overlap_bars

    ret_a = np.log(aligned.iloc[:, 0] / aligned.iloc[:, 0].shift(1)).dropna()
    ret_b = np.log(aligned.iloc[:, 1] / aligned.iloc[:, 1].shift(1)).dropna()
    rets = pd.concat([ret_a, ret_b], axis=1, join="inner").dropna()
    if len(rets) < 2:
        return None, None, overlap_bars

    full_corr = float(rets.iloc[:, 0].corr(rets.iloc[:, 1]))
    rolling = rets.iloc[:, 0].rolling(roll_window).corr(rets.iloc[:, 1]).dropna()
    rolling_corr_median = float(rolling.median()) if len(rolling) else None
    return full_corr, rolling_corr_median, overlap_bars


# Per-asset-class correlation thresholds for greedy_group_pairs. A pair's
# effective threshold is the STRICTER (max) of its two symbols' class
# thresholds; "default" is used for any asset_class not otherwise listed.
MIN_ABS_CORR = {"crypto": 0.75, "default": 0.6}


def _resolve_min_abs_corr(min_abs_corr, class_a, class_b) -> float:
    """Resolve the effective threshold for a pair given its two symbols'
    asset classes. `min_abs_corr` may be a plain float (uniform threshold,
    unchanged legacy behavior) or a dict mapping asset_class -> threshold
    with a "default" key for unlisted classes. The stricter (max) of the two
    per-class thresholds applies."""
    if isinstance(min_abs_corr, dict):
        default = min_abs_corr.get("default", 0.6)
        t_a = min_abs_corr.get(class_a, default)
        t_b = min_abs_corr.get(class_b, default)
        return max(t_a, t_b)
    return min_abs_corr


def greedy_group_pairs(pairs: list, min_abs_corr=0.6, max_group_size=4):
    """
    Greedy adjacency-based clustering with overlapping membership.

    Algorithm: sort pairs by |corr| descending, then for each passing pair:
      1. If some existing group already contains BOTH symbols, just append the
         corr to that group's corrs (no membership change).
      2. Elif some existing group contains exactly ONE of the two symbols and
         has capacity (< max_group_size), add the missing symbol to it and
         append the corr. If several groups qualify, the one with the highest
         mean |corr| wins (ties broken by lowest group index, i.e. the group
         seeded by the strongest pair) - keeps the strongest baskets cohesive
         and is deterministic.
      3. Else create a NEW group {a, b}. This guarantees every passing pair is
         represented in some group; symbols may therefore appear in multiple
         (overlapping) groups.

    Not guaranteed optimal, but deterministic, cheap, and good enough for
    generating candidate trading baskets.

    `pairs` is a list of dicts with keys: symbol_a, symbol_b, asset_class, full_corr,
    and optionally class_a/class_b (the two symbols' own asset classes; for a
    same-class pair these both equal `asset_class`, for a "cross" pair they
    hold the two distinct classes). When class_a/class_b are absent, both
    fall back to `asset_class` (backward compatible with older pair dicts).
    `asset_class` is the pair's own class, or "cross" for a cross-asset-class
    pair. A group's asset_class flips to "cross" when its membership spans
    classes (a "cross" pair seeds/joins it, or a joining pair's class differs
    from the group's established class).

    `min_abs_corr` may be a plain float (uniform threshold, as before) or a
    dict mapping asset_class -> threshold with a "default" key; a pair's
    effective threshold is then the stricter (max) of its two symbols' class
    thresholds (see `_resolve_min_abs_corr`).

    Returns a list of dicts: {asset_class, symbols: [...], mean_intra_corr}.
    """
    strong = []
    for p in pairs:
        if p.get("full_corr") is None:
            continue
        class_a = p.get("class_a", p["asset_class"])
        class_b = p.get("class_b", p["asset_class"])
        threshold = _resolve_min_abs_corr(min_abs_corr, class_a, class_b)
        if abs(p["full_corr"]) >= threshold:
            strong.append(p)
    strong.sort(key=lambda p: abs(p["full_corr"]), reverse=True)

    groups = []  # list of dicts: {"asset_class":..., "symbols": set(), "corrs": []}

    def _mark_if_cross(gi, ac):
        """Flip a group to 'cross' when a pair joining it straddles asset classes,
        or when the pair's own class doesn't match the group's established class.
        Never downgrades an already-"cross" group back to a single class.
        """
        if ac == "cross" or groups[gi]["asset_class"] not in (ac, "cross"):
            groups[gi]["asset_class"] = "cross"

    for p in strong:
        a, b, ac, corr = p["symbol_a"], p["symbol_b"], p["asset_class"], p["full_corr"]

        # Rule 1: a group already contains both symbols.
        both = [gi for gi, g in enumerate(groups) if a in g["symbols"] and b in g["symbols"]]
        if both:
            groups[both[0]]["corrs"].append(corr)
            _mark_if_cross(both[0], ac)
            continue

        # Rule 2: a group contains exactly one symbol and has capacity.
        candidates = [
            gi for gi, g in enumerate(groups)
            if len({a, b} & g["symbols"]) == 1 and len(g["symbols"]) < max_group_size
        ]
        if candidates:
            gi = max(
                candidates,
                key=lambda i: (
                    float(np.mean([abs(c) for c in groups[i]["corrs"]])) if groups[i]["corrs"] else 0.0,
                    -i,
                ),
            )
            missing = b if a in groups[gi]["symbols"] else a
            groups[gi]["symbols"].add(missing)
            groups[gi]["corrs"].append(corr)
            _mark_if_cross(gi, ac)
            continue

        # Rule 3: new group. Happens for a fresh pair, or when both symbols
        # sit in different full/ineligible groups (never drop a passing pair).
        groups.append({"asset_class": ac, "symbols": {a, b}, "corrs": [corr]})

    result = []
    for g in groups:
        if len(g["symbols"]) < 2:
            continue
        result.append({
            "asset_class": g["asset_class"],
            "symbols": sorted(g["symbols"]),
            "mean_intra_corr": float(np.mean(g["corrs"])) if g["corrs"] else None,
        })
    return result


def run_stage_correlation(conn, asset_class_filter=None, interval="1d", min_abs_corr=None):
    run_id = start_run(conn, "correlation", interval, {"asset_class_filter": asset_class_filter})

    # Latest passing universe survivors (most recent universe run only).
    q = """
        SELECT symbol, asset_class FROM universe_screen
        WHERE passed = 1 AND run_id = (SELECT MAX(run_id) FROM universe_screen)
    """
    params = []
    if asset_class_filter:
        q += " AND asset_class = ?"
        params.append(asset_class_filter)
    survivors = conn.execute(q, params).fetchall()

    if not survivors:
        print("Stage 2 (correlation): no passing universe survivors found. Run --stage universe first.")
        conn.commit()
        return run_id

    from kairos_strategies import calendar_days_for_bars

    price_cache.configure(remote=False)

    # Compute calendar window for the bar interval
    # For 1d, we need 400 days of data; for other intervals, scale accordingly.
    # bars_per_day for 1d = 1, so 400 bars = 400 days.
    bars_per_day = {
        "1m": 1440, "2m": 720, "5m": 288, "15m": 96, "30m": 48,
        "60m": 24, "90m": 16, "1h": 24, "1d": 1, "5d": 0.2,
        "1wk": 1 / 7, "1mo": 1 / 30, "3mo": 1 / 90,
    }.get(interval, 1)

    bars_needed = 400
    days_needed = calendar_days_for_bars(bars_needed, bars_per_day, "BTC-USD", buffer_days=0)

    end_dt = date.today()
    start_dt = end_dt - timedelta(days=days_needed)

    closes = {}
    classes = {}
    for symbol, ac in survivors:
        try:
            df = price_cache.get_price_data(
                symbol, start_date=start_dt.isoformat(), end_date=end_dt.isoformat(), interval=interval
            )
            if df is None or df.empty:
                continue
            df = df.sort_index().copy()
            df.columns = [c.lower() for c in df.columns]
            idx = pd.to_datetime(df.index)
            df.index = idx.tz_convert(None) if idx.tz is not None else idx
            closes[symbol] = df["close"].astype(float)
            classes[symbol] = ac
        except Exception as exc:
            print(f"  [warn] correlation fetch failed for {symbol}: {exc}")

    symbols = sorted(closes.keys())
    inserted_pairs = []
    pairs_for_grouping = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            a, b = symbols[i], symbols[j]
            pair_class = classes[a] if classes[a] == classes[b] else "cross"
            full_corr, rolling_median, overlap = compute_pair_correlation(closes[a], closes[b])
            if full_corr is None:
                continue
            row = {
                "symbol_a": a, "symbol_b": b, "asset_class": pair_class,
                "full_corr": full_corr, "rolling_corr_median": rolling_median,
                "overlap_bars": overlap,
            }
            insert_correlation_row(conn, run_id, row)
            inserted_pairs.append({"run_id": run_id, **row})
            # class_a/class_b carry the two symbols' own classes so
            # greedy_group_pairs can resolve a per-pair threshold even for
            # "cross" pairs (whose asset_class alone doesn't say which two
            # classes are involved). Kept out of the DB row/CSV export.
            pairs_for_grouping.append({**row, "class_a": classes[a], "class_b": classes[b]})

    effective_threshold = min_abs_corr if min_abs_corr is not None else MIN_ABS_CORR
    # Greedy clustering per asset class.
    groups = greedy_group_pairs(pairs_for_grouping, min_abs_corr=effective_threshold)
    inserted_groups = []
    gid = 0
    grouped_symbols = set()
    for g in groups:
        gid += 1
        row = {
            "group_id": gid, "asset_class": g["asset_class"],
            "symbols": ",".join(g["symbols"]), "mean_intra_corr": g["mean_intra_corr"],
        }
        insert_group_row(conn, run_id, row)
        inserted_groups.append({"run_id": run_id, **row})
        grouped_symbols.update(g["symbols"])

    # Singleton groups: survivors that had price data fetched but did not
    # land in any multi-symbol group (no peer correlated >= threshold).
    # This lets --stage auto cover them too, since it iterates suggested_groups.
    singleton_rows = []
    for symbol in symbols:
        if symbol in grouped_symbols:
            continue
        gid += 1
        row = {
            "group_id": gid, "asset_class": classes[symbol],
            "symbols": symbol, "mean_intra_corr": None,
        }
        insert_group_row(conn, run_id, row)
        inserted_groups.append({"run_id": run_id, **row})
        singleton_rows.append(row)

    conn.commit()
    csv_pairs = dump_csv("correlation_pairs", inserted_pairs, "correlation")
    csv_groups = dump_csv("suggested_groups", inserted_groups, "correlation")
    if isinstance(effective_threshold, dict):
        threshold_str = ", ".join(f"{k}={v}" for k, v in effective_threshold.items())
    else:
        threshold_str = str(effective_threshold)
    print(f"Effective min_abs_corr thresholds: {threshold_str}")
    print(f"\nStage 2 (correlation) done: {len(inserted_pairs)} pairs, "
          f"{len(inserted_groups)} suggested groups. run_id={run_id}.")
    print(f"CSV: {csv_pairs}, {csv_groups}")
    singleton_ids = {r["group_id"] for r in singleton_rows}
    for g in inserted_groups:
        if g["group_id"] in singleton_ids:
            print(f"  group {g['group_id']} [{g['asset_class']}] [singleton]: {g['symbols']}")
        else:
            print(f"  group {g['group_id']} [{g['asset_class']}]: {g['symbols']} "
                  f"(mean_corr={g['mean_intra_corr']:.3f})")
    return run_id


# ── Subprocess runner shared by stages 3/4/5 ─────────────────────────────────

def run_backtest_subprocess(assets, interval="1d", backtest_period="6m",
                             no_prediction=False, model_path=None, pred_samples=100,
                             extra_env=None, no_disabled_filter=False):
    """
    Invoke strategy/kairos_strategies.py as a subprocess and return the parsed
    JSON export (summary, strategy_rankings, shadow_performance).

    Shared by stage 3 (oracle, no_prediction=True), stage 4 (base model,
    no_prediction=False, model_path=None) and stage 5 (finetuned model,
    no_prediction=False, model_path=<checkpoint>). kairos_strategies.py
    already exposes a `--model` flag for a local finetuned checkpoint path,
    so stage 5 reuses it rather than inventing a new `--model_path` flag.

    `no_disabled_filter=True` appends `--no_disabled_filter`, bypassing the
    DB/class disabled-strategy resolution so every strategy is evaluated.
    The oracle stage (stage 3) always passes this - disabled strategies must
    still be evaluated by the oracle so they can be re-enabled once their
    numbers improve; base/finetuned (stages 4/5) never do, since they should
    keep skipping strategies that are already known to be disabled.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "uv", "run", STRATEGIES_SCRIPT,
        "--interval", interval,
        "--backtest_period", backtest_period,
        "--pred_samples", str(pred_samples),
        "--assets", *assets,
        "--export_json", tmp_path,
    ]
    if no_prediction:
        cmd.append("--no-prediction")
    if model_path:
        cmd.extend(["--model", model_path])
    if no_disabled_filter:
        cmd.append("--no_disabled_filter")

    env = None
    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)

    print(f"  [subprocess] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
    print(proc.stdout)
    if proc.returncode == 75:
        # EX_TEMPFAIL: kairos_gpu.ensure_cuda() healed the GPU but this
        # subprocess's cached torch state was stale. Retry exactly once -
        # a fresh subprocess will see the healed GPU.
        print(proc.stderr)
        print("  [subprocess] exit 75 (GPU recovered, retrying once in a fresh process)")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"kairos_strategies.py subprocess failed with code {proc.returncode}")

    with open(tmp_path) as f:
        payload = json.load(f)
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return payload


def _rows_from_export(payload: dict, assets, interval, backtest_period, stage: str):
    """Flatten export_json payload into one row per strategy."""
    shadow = payload.get("shadow_performance", {})
    rankings = dict(payload.get("strategy_rankings", []))
    rows = []
    for sname, sdata in shadow.items():
        pnl_list = sdata.get("pnl_list", [])
        wins = sum(1 for p in pnl_list if p > 0)
        win_rate = wins / len(pnl_list) if pnl_list else 0.0
        avg_pnl = float(np.mean(pnl_list)) if pnl_list else 0.0
        rows.append({
            "stage": stage,
            "strategy_name": sname,
            "sharpe": sdata.get("sharpe", rankings.get(sname)),
            "signal_count": sdata.get("signal_count", len(pnl_list)),
            "win_rate": win_rate,
            "avg_pnl_per_trade": avg_pnl,
            "assets": ",".join(assets),
            "interval": interval,
            "backtest_period": backtest_period,
        })
    return rows


def run_stage_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100,
                      disable_min_signals=5):
    run_id = start_run(conn, "oracle", interval, {
        "assets": assets, "backtest_period": backtest_period, "pred_samples": pred_samples,
    })
    payload = run_backtest_subprocess(
        assets, interval=interval, backtest_period=backtest_period,
        no_prediction=True, model_path=None, pred_samples=pred_samples,
        no_disabled_filter=True,
    )
    rows = _rows_from_export(payload, assets, interval, backtest_period, stage="oracle")
    for row in rows:
        insert_oracle_row(conn, run_id, row)
    conn.commit()

    csv_path = dump_csv("oracle_results", [{"run_id": run_id, **r} for r in rows], "oracle")
    rows_sorted = sorted(rows, key=lambda r: (r["sharpe"] if r["sharpe"] is not None else 0.0))
    negative = [r for r in rows_sorted if (r["sharpe"] or 0.0) < 0]
    build_stats = payload.get("strategy_build_stats") or {}
    if build_stats:
        print(
            f"\nStage 3 (oracle) done: built {build_stats.get('total_constructed', '?')}, "
            f"disabled {build_stats.get('disabled_removed', '?')}, "
            f"evaluating {build_stats.get('evaluated', '?')} strategies "
            f"({len(rows)} fired at least one signal). run_id={run_id}. CSV: {csv_path}"
        )
    else:
        print(f"\nStage 3 (oracle) done: {len(rows)} strategies. run_id={run_id}. CSV: {csv_path}")
    print(f"Strategies with negative Sharpe ({len(negative)}):")
    for r in negative:
        print(f"  {r['strategy_name']:<28} sharpe={r['sharpe']:.3f} n={r['signal_count']}")

    newly_disabled, re_enabled = refresh_disabled_strategies(
        conn, run_id, rows, interval, assets, min_signals=disable_min_signals,
    )
    print(
        f"[disabled] +{len(newly_disabled)} newly disabled: {newly_disabled}; "
        f"{len(re_enabled)} re-enabled: {re_enabled}"
    )
    current_rows = [
        dict(zip(
            ("interval", "assets", "strategy_name", "avg_pnl_per_trade", "sharpe",
             "signal_count", "source_run_id", "updated_at"),
            r,
        ))
        for r in conn.execute(
            "SELECT interval, assets, strategy_name, avg_pnl_per_trade, sharpe, "
            "signal_count, source_run_id, updated_at FROM disabled_strategies "
            "WHERE interval=? AND assets=?",
            (interval, ",".join(sorted(assets))),
        ).fetchall()
    ]
    dump_csv("disabled_strategies", current_rows, "oracle_disabled_strategies")
    return run_id


def run_stage_model(conn, stage, assets, interval="1d", backtest_period="6m",
                     pred_samples=100, model_path=None, extra_env=None):
    """
    Shared implementation for stage 4 ('base') and stage 5 ('finetuned').
    Not executed in this environment (needs GPU + downloaded/finetuned Kronos
    weights) but fully wired: parameterizes run_backtest_subprocess with
    no_prediction=False and (for 'finetuned') a --model checkpoint path.

    `extra_env` (e.g. KAIROS_PRED_CACHE_DIR) is passed through to the
    subprocess so --stage auto's per-run prediction cache is shared across
    the group subprocesses it spawns.
    """
    assert stage in ("base", "finetuned")
    run_id = start_run(conn, stage, interval, {
        "assets": assets, "backtest_period": backtest_period,
        "pred_samples": pred_samples, "model_path": model_path,
    })
    payload = run_backtest_subprocess(
        assets, interval=interval, backtest_period=backtest_period,
        no_prediction=False, model_path=model_path, pred_samples=pred_samples,
        extra_env=extra_env,
    )
    rows = _rows_from_export(payload, assets, interval, backtest_period, stage=stage)
    for row in rows:
        row["model_path"] = model_path
        insert_model_row(conn, run_id, row)
    conn.commit()

    csv_path = dump_csv("model_results", [{"run_id": run_id, **r} for r in rows], stage)
    build_stats = payload.get("strategy_build_stats") or {}
    if build_stats:
        print(
            f"\nStage {stage} done: built {build_stats.get('total_constructed', '?')}, "
            f"disabled {build_stats.get('disabled_removed', '?')}, "
            f"evaluating {build_stats.get('evaluated', '?')} strategies "
            f"({len(rows)} fired at least one signal). run_id={run_id}. CSV: {csv_path}"
        )
    else:
        print(f"\nStage {stage} done: {len(rows)} strategies. run_id={run_id}. CSV: {csv_path}")
    return run_id


def run_stage_rebuild_disabled(conn, min_signals=5):
    """
    Recompute the ENTIRE disabled_strategies table from scratch, DB-wide.

    Unlike run_stage_oracle's incremental refresh (scoped to the single
    profile it just tested), this walks every (interval, assets) profile
    present in oracle_results and rebuilds its disabled set from the most
    recent oracle run for that profile - the same "latest row per key"
    correlated-subquery pattern used by build_viability_report, without the
    interval/backtest_period WHERE filter (this rebuilds across ALL of them).
    Useful after manual DB edits, a criterion change (different
    --disable_min_signals), or to reconcile disabled_strategies with data
    written by direct (non --stage auto) CLI oracle invocations.

    Takes no --assets/--group_id - it operates DB-wide, grouping the latest
    oracle_results rows by (interval, assets) and calling
    refresh_disabled_strategies once per profile.

    Returns (profiles_processed, total_disabled) for the CLI summary print.
    """
    latest_q = """
        SELECT strategy_name, assets, interval, backtest_period,
               avg_pnl_per_trade, sharpe, signal_count, run_id
        FROM oracle_results
        WHERE stage = 'oracle'
        AND run_id = (
            SELECT MAX(run_id) FROM oracle_results o2
            WHERE o2.strategy_name = oracle_results.strategy_name
              AND o2.assets = oracle_results.assets
              AND o2.interval = oracle_results.interval
              AND o2.backtest_period = oracle_results.backtest_period
              AND o2.stage = 'oracle'
        )
    """
    latest_rows = conn.execute(latest_q).fetchall()

    # disabled_strategies has no backtest_period column, so within a
    # (interval, assets, strategy_name) triple we keep only the row from the
    # most recent run_id across all backtest_periods - "latest run wins".
    # Normalize `assets` to a sorted key here (not just inside
    # refresh_disabled_strategies): oracle_results.assets may hold both sort
    # orderings of the same profile (e.g. direct CLI calls that didn't sort
    # --assets), and this is precisely the stage meant to reconcile that -
    # without normalizing before grouping, the two literal orderings would be
    # treated as separate profiles, each independently DELETE+INSERTing into
    # the same normalized disabled_strategies row, silently discarding
    # whichever ordering's contribution was processed first.
    best = {}
    for strategy_name, assets, interval, backtest_period, avg_pnl, sharpe, signal_count, run_id in latest_rows:
        assets_norm = ",".join(sorted(assets.split(",")))
        key = (interval, assets_norm, strategy_name)
        if key not in best or run_id > best[key]["run_id"]:
            best[key] = {
                "strategy_name": strategy_name, "avg_pnl_per_trade": avg_pnl,
                "sharpe": sharpe, "signal_count": signal_count, "run_id": run_id,
            }

    profiles = {}
    for (interval, assets, strategy_name), row in best.items():
        profiles.setdefault((interval, assets), []).append(row)

    total_disabled = 0
    for (interval, assets_key), rows in profiles.items():
        assets_list = assets_key.split(",")
        source_run_id = max((r["run_id"] for r in rows), default=None)
        newly_disabled, re_enabled = refresh_disabled_strategies(
            conn, source_run_id, rows, interval, assets_list, min_signals=min_signals,
        )
        n = conn.execute(
            "SELECT COUNT(*) FROM disabled_strategies WHERE interval=? AND assets=?",
            (interval, ",".join(sorted(assets_list))),
        ).fetchone()[0]
        total_disabled += n
        print(f"  [{interval}] {assets_key}: +{len(newly_disabled)} disabled, "
              f"{len(re_enabled)} re-enabled (now {n} disabled)")

    print(f"\nrebuild_disabled done: {len(profiles)} profiles processed, "
          f"{total_disabled} strategies disabled across all profiles.")
    return len(profiles), total_disabled


# ── Automated finetuning (--stage finetune_next) ────────────────────────────

FINETUNE_BASE_MODEL = "NeoQuasar/Kronos-base"

# Yahoo Finance hard limits by interval (days of history available) - mirrors
# kairos_strategies.fetch_data_raw's yf_max_days table.
_YF_MAX_DAYS = {
    "1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 729, "90m": 60, "1h": 729,
}
_YF_MAX_DAYS_DEFAULT = 5 * 365


def finetune_model_dir(interval, assets) -> str:
    """
    Compute the on-disk directory for a finetuned model keyed by (interval, assets).

    `assets` may be a list of symbols or a comma-separated CSV string. Symbols
    are sorted and joined with underscores; commas become underscores, but
    '-' and '=' (e.g. BTC-USD, EURUSD=X) are kept as-is.

    Returns models/finetuned/{interval}__{SORTED_ASSETS_JOINED_BY_UNDERSCORE}/
    e.g. models/finetuned/1h__BTC-USD_ETH-USD_SOL-USD/
    """
    assets_list = assets.split(",") if isinstance(assets, str) else list(assets)
    sanitized = "_".join(sorted(assets_list))
    return os.path.join(MODELS_DIR, "finetuned", f"{interval}__{sanitized}")


def _period_to_days(period: str) -> int:
    """Convert a human period string (e.g. '6m', '1y') to calendar days.

    Uses the same unit definitions as kairos_strategies._period_to_bars
    (30-day months, 365-day years) for consistency with the rest of the
    pipeline's bar-count math.
    """
    n, unit = _parse_period(period)
    return {"d": n, "w": n * 7, "m": n * 30, "y": n * 365}[unit]


def compute_finetune_periods(backtest_period: str, interval: str, now=None) -> dict:
    """
    Compute train/test period boundaries (YYYY-MM-DD strings) for an
    automated finetune run.

    test_end = now; test_start = now - period_days(backtest_period) - the
    same window length as the base run being compared against, re-anchored
    to "now" (the pipeline doesn't track a base run's exact calendar dates,
    only its period label, so an exact-window replay isn't possible - see
    Out of scope in the plan).

    train_end = test_start (no leakage; the model's inference lookback
    overlapping the train tail is standard walk-forward and acceptable).

    train_start = train_end minus the data-source horizon for this interval
    (5 years for daily-ish intervals, capped at 729 days for 1h/60m, mirroring
    kairos_strategies.fetch_data_raw's yf_max_days) - the trainer's fetch
    just gets whatever history actually exists within that window.

    Returns {"train_start", "train_end", "test_start", "test_end"}.
    """
    if now is None:
        now = datetime.now()
    days = _period_to_days(backtest_period)
    test_end = now
    test_start = now - timedelta(days=days)
    train_end = test_start
    horizon_days = _YF_MAX_DAYS.get(interval, _YF_MAX_DAYS_DEFAULT)
    train_start = train_end - timedelta(days=horizon_days)

    return {
        "train_start": train_start.date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "test_start": test_start.date().isoformat(),
        "test_end": test_end.date().isoformat(),
    }


def select_finetune_candidate(conn, min_signals=3):
    """
    Select the top-ranked (assets, interval, backtest_period) profile for
    automated finetuning.

    Score = count of that profile's latest oracle strategies with
    sharpe > 0 AND signal_count >= min_signals ("viable-bar"); ties broken by
    the mean sharpe of those same viable-bar strategies. A profile is only a
    candidate if:
      - a stage='base' model_results run exists for the identical
        (assets, interval, backtest_period) triple (raw assets string, not
        sorted - the finetuned backtest must be comparable to something), and
      - it is not already present in finetuned_models under ANY status
        (training/accepted/rejected/failed), matched on the sorted-assets key -
        rejected/failed profiles are excluded permanently; manual re-queue
        (run_stage_finetune_next with explicit assets/interval) is the only
        way to retry one.

    Returns a dict {assets_raw, assets_sorted, interval, backtest_period,
    viable_count, mean_sharpe} for the top candidate, or None if none qualify.
    """
    latest_q = """
        SELECT strategy_name, assets, interval, backtest_period, sharpe, signal_count
        FROM oracle_results
        WHERE stage = 'oracle'
        AND run_id = (
            SELECT MAX(run_id) FROM oracle_results o2
            WHERE o2.strategy_name = oracle_results.strategy_name
              AND o2.assets = oracle_results.assets
              AND o2.interval = oracle_results.interval
              AND o2.backtest_period = oracle_results.backtest_period
              AND o2.stage = 'oracle'
        )
    """
    rows = conn.execute(latest_q).fetchall()

    profiles = {}
    for strategy_name, assets, interval, backtest_period, sharpe, signal_count in rows:
        key = (assets, interval, backtest_period)
        profiles.setdefault(key, []).append((sharpe, signal_count))

    already_registered = {
        r[0] for r in conn.execute("SELECT assets FROM finetuned_models").fetchall()
    }

    candidates = []
    for (assets, interval, backtest_period), strat_rows in profiles.items():
        assets_sorted = ",".join(sorted(assets.split(",")))
        if assets_sorted in already_registered:
            continue

        base_exists = conn.execute(
            "SELECT 1 FROM model_results WHERE assets=? AND interval=? "
            "AND backtest_period=? AND stage='base' LIMIT 1",
            (assets, interval, backtest_period),
        ).fetchone()
        if not base_exists:
            continue

        viable = [
            s for s, n in strat_rows
            if s is not None and s > 0 and (n or 0) >= min_signals
        ]
        candidates.append({
            "assets_raw": assets,
            "assets_sorted": assets_sorted,
            "interval": interval,
            "backtest_period": backtest_period,
            "viable_count": len(viable),
            "mean_sharpe": float(np.mean(viable)) if viable else 0.0,
        })

    if not candidates:
        return None

    candidates.sort(key=lambda c: (c["viable_count"], c["mean_sharpe"]), reverse=True)
    return candidates[0]


def compare_finetuned_vs_base(conn, assets_raw, interval, backtest_period, min_signals=3):
    """
    Compare the latest stage='finetuned' backtest against the latest
    stage='base' backtest for the same (assets_raw, interval, backtest_period)
    profile - matched on the RAW assets string as stored in model_results
    (the same string used to invoke each subprocess), per the registry's
    sorted-key/raw-string split.

    viable-bar strategy = sharpe > 0 AND signal_count >= min_signals (same
    gate as select_finetune_candidate). Accept iff:
        ft_count > base_count
        OR (ft_count == base_count AND ft_mean > base_mean)

    Returns {base_run_id, base_count, base_mean, ft_run_id, ft_count, ft_mean,
    accepted}. If a stage has no matching rows, its run_id is None and its
    count/mean default to 0/0.0.
    """
    def _latest_stage_stats(stage):
        q = """
            SELECT sharpe, signal_count, run_id
            FROM model_results
            WHERE assets = ? AND interval = ? AND backtest_period = ? AND stage = ?
            AND run_id = (
                SELECT MAX(run_id) FROM model_results
                WHERE assets = ? AND interval = ? AND backtest_period = ? AND stage = ?
            )
        """
        rows = conn.execute(
            q, (assets_raw, interval, backtest_period, stage,
                assets_raw, interval, backtest_period, stage),
        ).fetchall()
        if not rows:
            return None, 0, 0.0
        run_id = rows[0][2]
        viable = [
            s for s, n, _ in rows
            if s is not None and s > 0 and (n or 0) >= min_signals
        ]
        return run_id, len(viable), (float(np.mean(viable)) if viable else 0.0)

    base_run_id, base_count, base_mean = _latest_stage_stats("base")
    ft_run_id, ft_count, ft_mean = _latest_stage_stats("finetuned")

    accepted = ft_count > base_count or (ft_count == base_count and ft_mean > base_mean)

    return {
        "base_run_id": base_run_id, "base_count": base_count, "base_mean": base_mean,
        "ft_run_id": ft_run_id, "ft_count": ft_count, "ft_mean": ft_mean,
        "accepted": accepted,
    }


def acquire_finetune_lock(lock_path=None):
    """
    Acquire the exclusive "one active finetune_next instance" lock via
    fcntl.flock (LOCK_EX | LOCK_NB) on `lock_path` (default
    REPO_ROOT/data/finetune_next.lock).

    On success, best-effort truncates the file and writes the current pid,
    then returns the open file object - the CALLER must keep a reference to
    it for the lifetime of the run. The lock is released the instant the fd
    is closed, or automatically by the kernel if the process dies for any
    reason (crash, SIGKILL, OOM). That's the whole point of using flock here
    instead of a pid file: a pid file can be left behind by a crashed
    process and must be staleness-checked by hand; flock cannot go stale.

    Returns None if another live process already holds the lock (raised as
    BlockingIOError/OSError from flock) - the caller should treat this as
    "another instance is already running" and exit cleanly.
    """
    path = lock_path or FINETUNE_LOCK_PATH
    fd = open(path, "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        fd.close()
        return None

    try:
        fd.seek(0)
        fd.truncate()
        fd.write(str(os.getpid()))
        fd.flush()
    except OSError:
        pass  # best-effort only; the lock itself is already held regardless

    return fd


def _sweep_orphaned_training_finetunes(conn):
    """
    Mark every finetuned_models row with status='training' as 'failed'.

    Called only while the finetune_next lock is held. Since this process
    holds the exclusive lock, any row still sitting in status='training' at
    this point cannot belong to a live run - it is by definition a leftover
    from a previous instance that crashed before it could update its own
    row. Left as 'training' it would silently and permanently block
    select_finetune_candidate from ever reconsidering that profile (which
    excludes assets present in finetuned_models under ANY status). Marking
    it 'failed' instead makes it eligible for a manual re-queue.
    """
    rows = conn.execute(
        "SELECT id, assets_raw, interval FROM finetuned_models WHERE status = 'training'"
    ).fetchall()
    for row_id, assets_raw, interval in rows:
        conn.execute("UPDATE finetuned_models SET status = 'failed' WHERE id = ?", (row_id,))
        print(f"[finetune_next] orphaned 'training' row id={row_id} "
              f"({assets_raw}@{interval}) marked failed (previous run crashed); "
              f"re-queue manually if wanted.")
    if rows:
        conn.commit()


def run_stage_finetune_next(conn, assets=None, interval="1d", backtest_period="6m",
                             pred_samples=100, ft_epochs=10, ft_batch_size=32,
                             min_signals=3, dry_run=False, lock_path=None):
    """
    Orchestrate one automated finetune-and-compare cycle:
      0. Unless dry_run, acquire the exclusive finetune_next lock
         (acquire_finetune_lock) first, before anything else. If another
         instance already holds it, print a message and return None
         immediately - no candidate selection, no registry writes. Once
         held, sweep any orphaned status='training' rows (leftovers from a
         crashed previous run - see _sweep_orphaned_training_finetunes) to
         'failed' before candidate selection.
      1. Select the top not-yet-finetuned candidate (select_finetune_candidate),
         or use the explicitly supplied (assets, interval) for a manual re-queue.
      2. Insert a finetuned_models row with status='training' (claims the
         UNIQUE(assets, interval) slot immediately).
      3. Train via `uv run finetune --model NeoQuasar/Kronos-base --symbol
         <raw assets...> --interval I --start TRAIN_START --end TRAIN_END
         --device cuda --epochs E --batch-size B --output-model DIR` as a
         subprocess. A non-zero exit marks the row 'failed' and stops (the
         model dir is kept for post-mortem).
      4. Backtest the trained checkpoint via run_stage_model(stage="finetuned"),
         parameter-identical (assets/interval/backtest_period/pred_samples) to
         the last base run for this profile.
      5. Compare via compare_finetuned_vs_base; accept iff ft_count > base_count
         OR (ft_count == base_count AND ft_mean > base_mean).
      6. Update the registry row (accepted/rejected + run ids + counts/means),
         write an empty REJECTED marker file on rejection, write
         <dir>/metadata.json mirroring the registry row, and print a verdict.

    Manual re-queue: pass `assets` (a list of raw symbols, in the same order
    used by the base run's model_results.assets) to bypass ranking entirely
    and force this exact profile. Any existing finetuned_models row for it
    (matched on the sorted-assets key) is deleted first - this is the only
    way to retry a previously rejected/failed profile.

    `dry_run=True` prints the selected candidate, computed train/test
    periods, and the planned training command, then returns immediately with
    zero side effects: no lock is taken, no orphan sweep runs, no registry
    row inserted, no directories created, no subprocess executed.

    `lock_path` overrides the default REPO_ROOT/data/finetune_next.lock path
    (mainly for tests).

    Returns the finetuned_models row id on a real run, or None if there was
    no candidate (nothing to do), another instance already holds the lock,
    or dry_run was set.
    """
    lock_file = None
    try:
        if not dry_run:
            lock_file = acquire_finetune_lock(lock_path)
            if lock_file is None:
                print("[finetune_next] another instance is already running "
                      "(data/finetune_next.lock held); exiting.")
                return None
            _sweep_orphaned_training_finetunes(conn)

        manual = assets is not None
        if manual:
            assets_list = list(assets)
            assets_raw = ",".join(assets_list)
            candidate = {
                "assets_raw": assets_raw,
                "assets_sorted": ",".join(sorted(assets_list)),
                "interval": interval,
                "backtest_period": backtest_period,
                "viable_count": None,
                "mean_sharpe": None,
            }
        else:
            candidate = select_finetune_candidate(conn, min_signals=min_signals)
            if candidate is None:
                print("[finetune_next] no candidates found (no unregistered profile with "
                      "both oracle data and an existing base run).")
                return None

        return _run_finetune_next_body(
            conn, candidate, manual, interval, backtest_period, pred_samples,
            ft_epochs, ft_batch_size, min_signals, dry_run,
        )
    finally:
        if lock_file is not None:
            lock_file.close()


def _run_finetune_next_body(conn, candidate, manual, interval, backtest_period,
                             pred_samples, ft_epochs, ft_batch_size, min_signals, dry_run):
    assets_raw = candidate["assets_raw"]
    assets_sorted = candidate["assets_sorted"]
    interval = candidate["interval"]
    backtest_period = candidate["backtest_period"]
    assets_list = assets_raw.split(",")

    periods = compute_finetune_periods(backtest_period, interval)
    model_dir = finetune_model_dir(interval, assets_list)
    best_model_path = os.path.join(model_dir, "best_model")

    train_cmd = [
        "uv", "run", "finetune",
        "--model", FINETUNE_BASE_MODEL,
        "--symbol", *assets_list,
        "--interval", interval,
        "--start", periods["train_start"],
        "--end", periods["train_end"],
        "--device", "cuda",
        "--epochs", str(ft_epochs),
        "--batch-size", str(ft_batch_size),
        "--output-model", model_dir,
    ]

    print(f"[finetune_next] candidate: assets={assets_raw!r} interval={interval} "
          f"backtest_period={backtest_period} viable_count={candidate['viable_count']} "
          f"mean_sharpe={candidate['mean_sharpe']}")
    print(f"[finetune_next] periods: {periods}")
    print(f"[finetune_next] planned training command: {' '.join(train_cmd)}")

    if dry_run:
        print("[finetune_next] --dry_run: no side effects.")
        return None

    if manual:
        conn.execute("DELETE FROM finetuned_models WHERE assets = ?", (assets_sorted,))
        conn.commit()

    os.makedirs(model_dir, exist_ok=True)

    row_id = insert_finetune_registry_row(conn, {
        "assets": assets_sorted, "assets_raw": assets_raw, "interval": interval,
        "backtest_period": backtest_period, "status": "training", **periods,
    })
    print(f"[finetune_next] registered id={row_id} status=training model_dir={model_dir}")

    def _write_metadata(extra: dict):
        meta = {
            "id": row_id, "assets": assets_sorted, "assets_raw": assets_raw,
            "interval": interval, "backtest_period": backtest_period, **periods,
        }
        meta.update(extra)
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

    proc = subprocess.run(train_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        update_finetune_registry_row(conn, row_id, status="failed")
        _write_metadata({"status": "failed", "model_path": None})
        print(f"\n[finetune_next] VERDICT: FAILED (training subprocess exit "
              f"{proc.returncode}). id={row_id}. Model dir kept for post-mortem: {model_dir}")
        return row_id

    update_finetune_registry_row(conn, row_id, model_path=best_model_path)

    ft_run_id = run_stage_model(
        conn, stage="finetuned", assets=assets_list, interval=interval,
        backtest_period=backtest_period, pred_samples=pred_samples,
        model_path=best_model_path,
    )

    comparison = compare_finetuned_vs_base(
        conn, assets_raw, interval, backtest_period, min_signals=min_signals,
    )
    status = "accepted" if comparison["accepted"] else "rejected"

    update_finetune_registry_row(
        conn, row_id, status=status,
        base_run_id=comparison["base_run_id"], finetuned_run_id=ft_run_id,
        base_viable_count=comparison["base_count"], ft_viable_count=comparison["ft_count"],
        base_mean_sharpe=comparison["base_mean"], ft_mean_sharpe=comparison["ft_mean"],
    )

    _write_metadata({
        "status": status, "model_path": best_model_path,
        "base_run_id": comparison["base_run_id"], "finetuned_run_id": ft_run_id,
        "base_viable_count": comparison["base_count"], "ft_viable_count": comparison["ft_count"],
        "base_mean_sharpe": comparison["base_mean"], "ft_mean_sharpe": comparison["ft_mean"],
    })

    if not comparison["accepted"]:
        open(os.path.join(model_dir, "REJECTED"), "w").close()

    print(f"\n[finetune_next] VERDICT: {status.upper()}")
    print(f"  assets={assets_raw} interval={interval} backtest_period={backtest_period}")
    print(f"  base: viable_count={comparison['base_count']} mean_sharpe={comparison['base_mean']:.4f} "
          f"(run_id={comparison['base_run_id']})")
    print(f"  ft:   viable_count={comparison['ft_count']} mean_sharpe={comparison['ft_mean']:.4f} "
          f"(run_id={comparison['ft_run_id']})")
    print(f"  model_path={best_model_path}")
    print(f"  registry id={row_id}")

    return row_id


# ── Viability report ─────────────────────────────────────────────────────────

def _get_metric_columns(conn, table_name):
    """Get metric column names from a results table, excluding identifying columns.

    Identifying columns: run_id, stage, strategy_name, assets, interval, backtest_period
    Returns a tuple: (actual_db_columns, report_column_names) where report names apply standard renaming
    (e.g., signal_count → signals for the report).
    """
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    all_cols = [row[1] for row in cursor.fetchall()]  # row[1] is the column name
    identifying = {"run_id", "stage", "strategy_name", "assets", "interval", "backtest_period"}
    metrics = [c for c in all_cols if c not in identifying]
    # Standard naming: signal_count → signals in the report
    report_names = [c.replace("signal_count", "signals") if c == "signal_count" else c for c in metrics]
    return metrics, report_names


def build_viability_report(conn, intervals, backtest_period, min_sharpe=0.0, min_signals=3,
                            asset_class_filter=None):
    """
    Build a viability report joining latest oracle and base results.

    For each interval and backtest_period, fetches the latest oracle_results and
    model_results (stage='base') rows per (strategy_name, assets, interval, backtest_period),
    performs an outer join on those keys, computes derived columns (asset_class,
    signals_per_week, viable), and returns a DataFrame with columns:
      strategy_name, assets, asset_class, interval, backtest_period,
      [oracle_<metric> for each metric in the results schema],
      [base_<metric> for each metric in the results schema],
      signals_per_week, viable

    Columns are ordered with identifying columns first, then per-side metrics (prefixed),
    then derived columns (signals_per_week, viable).

    Viable = (oracle_sharpe > min_sharpe) & (base_sharpe > min_sharpe) &
             (min(oracle_signals, base_signals) >= min_signals);
    NaN on either side → False.

    signals_per_week = base_signals / _period_to_weeks(backtest_period),
    falling back to oracle_signals if base is NaN.

    Sorted: viable first (descending), then base_sharpe descending.
    """
    # Get metric columns from the results tables (generic extraction from schema)
    oracle_db_cols, oracle_report_cols = _get_metric_columns(conn, "oracle_results")
    base_db_cols, base_report_cols = _get_metric_columns(conn, "model_results")

    all_rows = []

    for interval in intervals:
        # Build SELECT clause for oracle: strategy_name, assets, then all metrics (DB names), then run_id
        oracle_select = "strategy_name, assets, " + ", ".join(oracle_db_cols) + ", run_id"
        oracle_q = f"""
            SELECT {oracle_select}
            FROM oracle_results
            WHERE interval = ? AND backtest_period = ? AND stage = 'oracle'
            AND run_id = (
                SELECT MAX(run_id) FROM oracle_results o2
                WHERE o2.strategy_name = oracle_results.strategy_name
                  AND o2.assets = oracle_results.assets
                  AND o2.interval = oracle_results.interval
                  AND o2.backtest_period = oracle_results.backtest_period
                  AND o2.stage = 'oracle'
            )
        """
        oracle_rows = conn.execute(oracle_q, (interval, backtest_period)).fetchall()
        oracle_dict = {}
        for row in oracle_rows:
            key = (row[0], row[1])  # (strategy_name, assets)
            # Map metrics to prefixed report names, last item is run_id
            oracle_data = {f"oracle_{oracle_report_cols[i]}": row[2 + i] for i in range(len(oracle_report_cols))}
            oracle_data["oracle_run_id"] = row[2 + len(oracle_report_cols)]
            oracle_dict[key] = oracle_data

        # Build SELECT clause for base: strategy_name, assets, then all metrics (DB names), then run_id
        base_select = "strategy_name, assets, " + ", ".join(base_db_cols) + ", run_id"
        base_q = f"""
            SELECT {base_select}
            FROM model_results
            WHERE interval = ? AND backtest_period = ? AND stage = 'base'
            AND run_id = (
                SELECT MAX(run_id) FROM model_results m2
                WHERE m2.strategy_name = model_results.strategy_name
                  AND m2.assets = model_results.assets
                  AND m2.interval = model_results.interval
                  AND m2.backtest_period = model_results.backtest_period
                  AND m2.stage = 'base'
            )
        """
        base_rows = conn.execute(base_q, (interval, backtest_period)).fetchall()
        base_dict = {}
        for row in base_rows:
            key = (row[0], row[1])  # (strategy_name, assets)
            # Map metrics to prefixed report names, last item is run_id
            base_data = {f"base_{base_report_cols[i]}": row[2 + i] for i in range(len(base_report_cols))}
            base_data["base_run_id"] = row[2 + len(base_report_cols)]
            base_dict[key] = base_data

        # OUTER join: union of all keys
        all_keys = set(oracle_dict.keys()) | set(base_dict.keys())

        for strategy_name, assets in all_keys:
            oracle_data = oracle_dict.get((strategy_name, assets), {})
            base_data = base_dict.get((strategy_name, assets), {})

            # Build row with identifying columns and per-side metrics
            row = {
                "strategy_name": strategy_name,
                "assets": assets,
                "asset_class": asset_class_for(assets.split(",")),
                "interval": interval,
                "backtest_period": backtest_period,
            }
            row.update(oracle_data)
            row.update(base_data)

            # Compute signals_per_week: prefer base_signals, fall back to oracle_signals
            signals = row.get("base_signals")
            if pd.isna(signals) or signals is None:
                signals = row.get("oracle_signals")
            if pd.notna(signals) and signals is not None:
                row["signals_per_week"] = float(signals) / _period_to_weeks(backtest_period)
            else:
                row["signals_per_week"] = None

            # Compute viable flag
            oracle_sharpe = row.get("oracle_sharpe")
            base_sharpe = row.get("base_sharpe")
            oracle_sig = row.get("oracle_signals")
            base_sig = row.get("base_signals")

            viable = False
            if (pd.notna(oracle_sharpe) and oracle_sharpe > min_sharpe and
                pd.notna(base_sharpe) and base_sharpe > min_sharpe):
                min_sig_count = min(
                    oracle_sig if pd.notna(oracle_sig) else float('inf'),
                    base_sig if pd.notna(base_sig) else float('inf'),
                )
                if min_sig_count >= min_signals:
                    viable = True

            row["viable"] = viable

            # Apply asset_class_filter if provided
            if asset_class_filter is None or row["asset_class"] == asset_class_filter:
                all_rows.append(row)

    # Determine column order: identifying, oracle metrics, oracle_run_id, base metrics, base_run_id,
    # base_model_path (if exists), then derived columns
    identifying_cols = ["strategy_name", "assets", "asset_class", "interval", "backtest_period"]
    # Oracle: metrics then run_id
    oracle_cols = [f"oracle_{m}" for m in oracle_report_cols] + ["oracle_run_id"]
    # Base: metrics then run_id, then model_path (if exists)
    base_metric_cols = [f"base_{m}" for m in base_report_cols if m != "model_path"]
    base_cols = base_metric_cols + ["base_run_id"]
    if "model_path" in base_report_cols:
        base_cols.append("base_model_path")
    derived_cols = ["signals_per_week", "viable"]
    col_order = identifying_cols + oracle_cols + base_cols + derived_cols

    # Convert to DataFrame
    if not all_rows:
        df = pd.DataFrame(columns=col_order)
    else:
        df = pd.DataFrame(all_rows)
        # Reorder columns, keeping only those that exist
        col_order = [c for c in col_order if c in df.columns]
        df = df[col_order]

    # Sort: viable first (descending), then base_sharpe descending
    # Only sort by base_sharpe if it exists in the DataFrame (might not if no base results)
    if "base_sharpe" in df.columns:
        df = df.sort_values(
            by=["viable", "base_sharpe"],
            ascending=[False, False],
            na_position="last",
        )
    else:
        df = df.sort_values(
            by=["viable"],
            ascending=[False],
            na_position="last",
        )

    return df


def persist_viability_report(conn, df, run_id):
    """Persist viability report DataFrame to the viability_report table and CSV."""
    # Insert rows into table
    for _, row in df.iterrows():
        conn.execute(
            """INSERT INTO viability_report
               (run_id, strategy_name, assets, asset_class, interval, backtest_period,
                oracle_sharpe, oracle_signals, oracle_win_rate, oracle_avg_pnl_per_trade, oracle_run_id,
                base_sharpe, base_signals, base_win_rate, base_avg_pnl_per_trade, base_run_id, base_model_path,
                signals_per_week, viable)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, row["strategy_name"], row["assets"], row["asset_class"], row["interval"], row["backtest_period"],
             row.get("oracle_sharpe"), row.get("oracle_signals"), row.get("oracle_win_rate"),
             row.get("oracle_avg_pnl_per_trade"), row.get("oracle_run_id"),
             row.get("base_sharpe"), row.get("base_signals"), row.get("base_win_rate"),
             row.get("base_avg_pnl_per_trade"), row.get("base_run_id"), row.get("base_model_path"),
             row.get("signals_per_week"), int(row.get("viable", False))),
        )
    conn.commit()

    # Write CSV
    rows_for_csv = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        rows_for_csv.append(row_dict)

    csv_path = dump_csv("viability_report", rows_for_csv, "auto")
    return csv_path


# ── Auto stage: chained universe → correlation → per-group oracle → base ─────

def _print_report_summary(df, intervals):
    """Print viability report summary (shared by run_stage_auto and --report_only).

    Args:
        df: Viability report DataFrame
        intervals: List of interval strings (e.g., ["1d", "1h"])
    """
    viable_count = len(df[df["viable"]])
    total_count = len(df)
    interval_breakdown = {}
    for interval in intervals:
        interval_df = df[df["interval"] == interval]
        interval_viable = len(interval_df[interval_df["viable"]])
        interval_breakdown[interval] = f"{interval_viable}/{len(interval_df)}"

    breakdown_str = ", ".join([f"{i}: {interval_breakdown[i]}" for i in intervals])
    print(f"Viability report: {total_count} strategies, {viable_count} viable "
          f"(interval breakdown: {breakdown_str})")


def run_stage_auto(conn, intervals, backtest_period, asset_class_filter=None,
                   pred_samples=100, min_sharpe=0.0, min_signals=3, force=False,
                   skip_universe=False, min_abs_corr=None, disable_min_signals=5):
    """
    Chain universe → correlation → per-group oracle → per-group base for each interval.

    Implements resumability keyed on (assets_key, interval, backtest_period) + stage="base" for model_results.
    Per-group failure isolation with RuntimeError try/except.
    Returns DataFrame from build_viability_report.

    A per-run prediction cache directory is created and shared (via
    KAIROS_PRED_CACHE_DIR) with every base/finetuned subprocess spawned during
    this auto run, so identical per-bar Kronos predictions computed for one
    overlapping group are reused by later groups instead of recomputed. The
    directory is removed when the run finishes (success or failure).
    """
    params = {
        "intervals": intervals,
        "backtest_period": backtest_period,
        "asset_class_filter": asset_class_filter,
        "pred_samples": pred_samples,
        "min_sharpe": min_sharpe,
        "min_signals": min_signals,
        "force": force,
        "skip_universe": skip_universe,
    }
    auto_run_id = start_run(conn, "auto", None, params)

    failures = []  # Track (group_id, error_msg) for final summary

    cache_dir = tempfile.mkdtemp(prefix=f"kairos_predcache_run{auto_run_id}_")
    extra_env = {"KAIROS_PRED_CACHE_DIR": cache_dir}
    try:
        return _run_stage_auto_body(
            conn, intervals, backtest_period, asset_class_filter, pred_samples,
            min_sharpe, min_signals, force, skip_universe, min_abs_corr,
            auto_run_id, failures, extra_env, disable_min_signals,
        )
    finally:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)


def _run_stage_auto_body(conn, intervals, backtest_period, asset_class_filter,
                          pred_samples, min_sharpe, min_signals, force,
                          skip_universe, min_abs_corr, auto_run_id, failures, extra_env,
                          disable_min_signals=5):
    for interval in intervals:
        print(f"\n=== Auto stage for interval {interval} ===")

        # Step 1: Universe (skip if skip_universe and prior run exists)
        universe_run_id = None
        if skip_universe:
            # Check if universe run exists for this interval
            existing = conn.execute(
                "SELECT MAX(run_id) FROM runs WHERE stage='universe' AND interval=?",
                (interval,),
            ).fetchone()
            if existing and existing[0]:
                print(f"[skip] universe {interval} (existing run {existing[0]})")
                universe_run_id = existing[0]
            else:
                universe_run_id = run_stage_universe(conn, interval=interval)
        else:
            universe_run_id = run_stage_universe(conn, interval=interval)

        # Step 2: Correlation
        if skip_universe:
            # Check if correlation run exists for this interval
            existing = conn.execute(
                "SELECT MAX(run_id) FROM runs WHERE stage='correlation' AND interval=?",
                (interval,),
            ).fetchone()
            if existing and existing[0]:
                print(f"[skip] correlation {interval} (existing run {existing[0]})")
                correlation_run_id = existing[0]
            else:
                correlation_run_id = run_stage_correlation(conn, asset_class_filter=asset_class_filter,
                                                            interval=interval, min_abs_corr=min_abs_corr)
        else:
            correlation_run_id = run_stage_correlation(conn, asset_class_filter=asset_class_filter,
                                                        interval=interval, min_abs_corr=min_abs_corr)

        # Step 3: Fetch suggested_groups for the latest correlation run
        groups = conn.execute(
            "SELECT group_id, symbols FROM suggested_groups WHERE run_id = ? ORDER BY group_id",
            (correlation_run_id,),
        ).fetchall()

        if not groups:
            print(f"[warn] no suggested groups found for correlation run {correlation_run_id}")
            continue

        # Step 4: Per group: oracle then base
        for group_id, symbols_str in groups:
            assets = symbols_str.split(",")
            assets_key = ",".join(sorted(assets))

            print(f"\n  [group {group_id}] {assets_key}")

            # Check resumability and run oracle
            oracle_exists = conn.execute(
                "SELECT run_id FROM oracle_results WHERE assets=? AND interval=? "
                "AND backtest_period=? AND stage='oracle' LIMIT 1",
                (assets_key, interval, backtest_period),
            ).fetchone()

            if oracle_exists and not force:
                print(f"    [skip] oracle (run_id={oracle_exists[0]} exists)")
            else:
                try:
                    oracle_run_id = run_stage_oracle(conn, assets, interval=interval,
                                                     backtest_period=backtest_period,
                                                     pred_samples=pred_samples,
                                                     disable_min_signals=disable_min_signals)
                    print(f"    [done] oracle (run_id={oracle_run_id})")
                except Exception as exc:
                    failures.append({"group_id": group_id, "assets": assets_key,
                                   "stage": "oracle", "error": str(exc)})
                    print(f"    [fail] oracle: {exc}")
                    continue  # Skip base for this group

            # Check resumability and run base
            base_exists = conn.execute(
                "SELECT run_id FROM model_results WHERE assets=? AND interval=? "
                "AND backtest_period=? AND stage='base' LIMIT 1",
                (assets_key, interval, backtest_period),
            ).fetchone()

            if base_exists and not force:
                print(f"    [skip] base (run_id={base_exists[0]} exists)")
            else:
                try:
                    base_run_id = run_stage_model(conn, "base", assets, interval=interval,
                                                  backtest_period=backtest_period,
                                                  pred_samples=pred_samples, model_path=None,
                                                  extra_env=extra_env)
                    print(f"    [done] base (run_id={base_run_id})")
                except Exception as exc:
                    failures.append({"group_id": group_id, "assets": assets_key,
                                   "stage": "base", "error": str(exc)})
                    print(f"    [fail] base: {exc}")

    # Failure summary
    if failures:
        print(f"\n=== Failure summary ({len(failures)} groups failed) ===")
        for f in failures:
            print(f"  {f['stage']} {f['assets']}: {f['error']}")

    # Build and persist viability report
    print(f"\n=== Building viability report ===")
    df = build_viability_report(conn, intervals, backtest_period, min_sharpe=min_sharpe,
                                min_signals=min_signals, asset_class_filter=asset_class_filter)

    # Persist report to table and CSV
    csv_path = persist_viability_report(conn, df, auto_run_id)

    # Summary
    _print_report_summary(df, intervals)

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _group_symbols_from_db(conn, group_id):
    row = conn.execute(
        "SELECT symbols FROM suggested_groups WHERE group_id = ? "
        "AND run_id = (SELECT MAX(run_id) FROM suggested_groups) ",
        (group_id,),
    ).fetchone()
    if row is None:
        raise SystemExit(f"No suggested group with group_id={group_id} found. Run --stage correlation first.")
    return row[0].split(",")


def _parse_min_abs_corr(raw: list):
    """Parse --min_abs_corr CLI tokens into a float (uniform threshold) or a
    dict of {asset_class: threshold} (with "default" key). Raises
    argparse.ArgumentTypeError-compatible ValueError on malformed input."""
    if raw is None:
        return None
    if len(raw) == 1 and "=" not in raw[0]:
        try:
            return float(raw[0])
        except ValueError:
            raise ValueError(f"--min_abs_corr: invalid float value: {raw[0]!r}")
    result = {}
    for token in raw:
        if "=" not in token:
            raise ValueError(
                f"--min_abs_corr: expected 'class=value' tokens (or a single float), got {token!r}"
            )
        key, _, value = token.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--min_abs_corr: empty class name in {token!r}")
        try:
            result[key] = float(value)
        except ValueError:
            raise ValueError(f"--min_abs_corr: invalid float value in {token!r}")
    if "default" not in result:
        result["default"] = 0.6
    return result


def _build_parser():
    """Build and return the argparse parser (extracted for testability)."""
    parser = argparse.ArgumentParser(description="Kairos staged asset-discovery pipeline")
    parser.add_argument("--stage", required=True,
                         choices=["universe", "correlation", "oracle", "base", "finetuned", "auto",
                                  "rebuild_disabled", "finetune_next"])
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--intervals", nargs="+", default=None,
                         help="Bar intervals for --stage auto (e.g. 1d 1h)")
    parser.add_argument("--backtest_period", default="6m")
    parser.add_argument("--pred_samples", type=int, default=100)
    parser.add_argument("--assets", nargs="+", default=None, metavar="SYM")
    parser.add_argument("--group_id", type=int, default=None)
    parser.add_argument("--asset_class", default=None, choices=["crypto", "equity", "fx_commodity"],
                         help="Restrict --stage correlation to one asset class")
    parser.add_argument("--min_abs_corr", nargs="+", default=None, metavar="VALUE",
                         help="Correlation grouping threshold: either a single float "
                              "(--min_abs_corr 0.7, uniform threshold) or class=value pairs "
                              "(--min_abs_corr crypto=0.8 equity=0.65 default=0.6). "
                              "Valid with --stage correlation or --stage auto.")
    parser.add_argument("--min_sharpe", type=float, default=0.0,
                         help="Minimum Sharpe for viability (--stage auto only)")
    parser.add_argument("--min_signals", type=int, default=3,
                         help="Minimum signal count for viability (--stage auto only)")
    parser.add_argument("--disable_min_signals", type=int, default=5,
                         help="Minimum oracle signal_count for the disabled_strategies criterion "
                              "(--stage oracle/auto/rebuild_disabled only)")
    parser.add_argument("--force", action="store_true",
                         help="Re-run completed stages (--stage auto only)")
    parser.add_argument("--skip_universe", action="store_true",
                         help="Reuse existing universe/correlation runs (--stage auto only)")
    parser.add_argument("--report_only", action="store_true",
                         help="Skip execution; rebuild report from DB (--stage auto only)")
    # TODO: kairos_strategies.py has no dedicated --model_path flag; it reuses
    # --model for the local checkpoint path, which we forward as model_path here.
    parser.add_argument("--model_path", default=None,
                         help="Finetuned Kronos checkpoint path (stage=finetuned only)")
    parser.add_argument("--ft_epochs", type=int, default=10,
                         help="Training epochs for the finetune subprocess (--stage finetune_next only)")
    parser.add_argument("--ft_batch_size", type=int, default=32,
                         help="Training batch size for the finetune subprocess (--stage finetune_next only)")
    parser.add_argument("--dry_run", action="store_true",
                         help="Print the selected candidate/periods/planned command without "
                              "executing anything (--stage finetune_next only)")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Get the actual argv to check if flags were explicitly passed
    actual_argv = argv if argv is not None else sys.argv[1:]

    # Enforce flag exclusivity constraints
    # 1. --stage auto + --interval (singular, not default) → error
    if args.stage == "auto" and "--interval" in actual_argv:
        parser.error("--stage auto uses --intervals (plural), not --interval (singular)")

    # 2. --intervals (plural) + non-auto stage → error
    if args.intervals is not None and args.stage != "auto":
        parser.error("--intervals is only valid with --stage auto; use --interval for other stages")

    # 3. Auto-specific flags with non-auto stage → error
    auto_only_flags = ["min_sharpe", "min_signals", "force", "skip_universe", "report_only"]
    for flag_name in auto_only_flags:
        # Check both hyphenated and underscored versions
        flag_hyphen = f"--{flag_name.replace('_', '-')}"
        flag_underscore = f"--{flag_name}"
        if args.stage != "auto" and (flag_hyphen in actual_argv or flag_underscore in actual_argv):
            parser.error(f"{flag_hyphen} is only valid with --stage auto")

    if args.stage not in ("correlation", "auto") and args.min_abs_corr is not None:
        parser.error("--min_abs_corr is only valid with --stage correlation or --stage auto")

    disable_min_signals_flag = "--disable_min_signals" in actual_argv
    if args.stage not in ("oracle", "auto", "rebuild_disabled") and disable_min_signals_flag:
        parser.error("--disable_min_signals is only valid with --stage oracle, auto, or rebuild_disabled")

    finetune_next_only_flags = ["ft_epochs", "ft_batch_size", "dry_run"]
    for flag_name in finetune_next_only_flags:
        flag_hyphen = f"--{flag_name.replace('_', '-')}"
        flag_underscore = f"--{flag_name}"
        if args.stage != "finetune_next" and (flag_hyphen in actual_argv or flag_underscore in actual_argv):
            parser.error(f"{flag_underscore} is only valid with --stage finetune_next")

    try:
        min_abs_corr = _parse_min_abs_corr(args.min_abs_corr)
    except ValueError as exc:
        parser.error(str(exc))

    conn = get_connection()

    if args.stage == "auto":
        intervals = args.intervals if args.intervals else ["1d"]

        if args.report_only:
            # Skip execution, rebuild report from DB
            report_run_id = start_run(conn, "auto", ", ".join(intervals),
                                    {"report_only": True, "asset_class_filter": args.asset_class})
            df = build_viability_report(conn, intervals, args.backtest_period,
                                       min_sharpe=args.min_sharpe, min_signals=args.min_signals,
                                       asset_class_filter=args.asset_class)
            csv_path = persist_viability_report(conn, df, report_run_id)
            _print_report_summary(df, intervals)
            print(f"CSV: {csv_path}")
        else:
            # Full auto pipeline execution
            run_stage_auto(conn, intervals, args.backtest_period,
                          asset_class_filter=args.asset_class, pred_samples=args.pred_samples,
                          min_sharpe=args.min_sharpe, min_signals=args.min_signals,
                          force=args.force, skip_universe=args.skip_universe,
                          min_abs_corr=min_abs_corr, disable_min_signals=args.disable_min_signals)

    elif args.stage == "universe":
        run_stage_universe(conn, interval=args.interval)
    elif args.stage == "correlation":
        run_stage_correlation(conn, asset_class_filter=args.asset_class, interval=args.interval,
                               min_abs_corr=min_abs_corr)
    elif args.stage == "oracle":
        assets = args.assets
        if args.group_id is not None:
            assets = _group_symbols_from_db(conn, args.group_id)
        if not assets:
            raise SystemExit("--stage oracle requires --assets SYM... or --group_id N")
        run_stage_oracle(conn, assets, interval=args.interval,
                          backtest_period=args.backtest_period, pred_samples=args.pred_samples,
                          disable_min_signals=args.disable_min_signals)
    elif args.stage in ("base", "finetuned"):
        assets = args.assets
        if args.group_id is not None:
            assets = _group_symbols_from_db(conn, args.group_id)
        if not assets:
            raise SystemExit(f"--stage {args.stage} requires --assets SYM... or --group_id N")
        run_stage_model(conn, args.stage, assets, interval=args.interval,
                         backtest_period=args.backtest_period, pred_samples=args.pred_samples,
                         model_path=args.model_path if args.stage == "finetuned" else None)
    elif args.stage == "rebuild_disabled":
        run_stage_rebuild_disabled(conn, min_signals=args.disable_min_signals)
    elif args.stage == "finetune_next":
        run_stage_finetune_next(
            conn, assets=args.assets, interval=args.interval,
            backtest_period=args.backtest_period, pred_samples=args.pred_samples,
            ft_epochs=args.ft_epochs, ft_batch_size=args.ft_batch_size,
            min_signals=args.min_signals, dry_run=args.dry_run,
        )

    conn.close()


if __name__ == "__main__":
    main()
