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
import json
import sqlite3
import subprocess
import sys as _sys
import tempfile
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd

import price_cache
from kairos_strategies import asset_class_for, _period_to_weeks

# ── Paths ──────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "data", "pipeline_results.db")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
STRATEGIES_SCRIPT = os.path.join(REPO_ROOT, "strategy", "kairos_strategies.py")

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
    fieldnames = list(rows[0].keys())
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


def greedy_group_pairs(pairs: list, min_abs_corr=0.6, max_group_size=4):
    """
    Greedy adjacency-based clustering.

    Algorithm choice: sort pairs by |corr| descending, then greedily add each
    pair's two symbols into a shared group as long as (a) both symbols'
    existing group (if any) allow room, and (b) merging would not exceed
    max_group_size. This is a simple greedy union-find-like approach - not
    guaranteed optimal, but deterministic, cheap, and good enough for
    generating a handful of candidate trading baskets per asset class.

    `pairs` is a list of dicts with keys: symbol_a, symbol_b, asset_class, full_corr.
    Returns a list of dicts: {asset_class, symbols: [...], mean_intra_corr}.
    """
    strong = [p for p in pairs if p.get("full_corr") is not None and abs(p["full_corr"]) >= min_abs_corr]
    strong.sort(key=lambda p: abs(p["full_corr"]), reverse=True)

    # symbol -> group index (int)
    symbol_to_group = {}
    groups = []  # list of dicts: {"asset_class":..., "symbols": set(), "corrs": []}

    for p in strong:
        a, b, ac, corr = p["symbol_a"], p["symbol_b"], p["asset_class"], p["full_corr"]
        ga = symbol_to_group.get(a)
        gb = symbol_to_group.get(b)

        if ga is None and gb is None:
            groups.append({"asset_class": ac, "symbols": {a, b}, "corrs": [corr]})
            gi = len(groups) - 1
            symbol_to_group[a] = gi
            symbol_to_group[b] = gi
        elif ga is not None and gb is None:
            if len(groups[ga]["symbols"]) < max_group_size:
                groups[ga]["symbols"].add(b)
                groups[ga]["corrs"].append(corr)
                symbol_to_group[b] = ga
        elif gb is not None and ga is None:
            if len(groups[gb]["symbols"]) < max_group_size:
                groups[gb]["symbols"].add(a)
                groups[gb]["corrs"].append(corr)
                symbol_to_group[a] = gb
        else:
            if ga == gb:
                groups[ga]["corrs"].append(corr)
            # Different existing groups: do not merge (keeps algorithm simple
            # and avoids unbounded group growth / overlap ambiguity).

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


def run_stage_correlation(conn, asset_class_filter=None):
    interval = "1d"
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

    price_cache.configure(remote=False)
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=400)

    closes = {}
    classes = {}
    for symbol, ac in survivors:
        try:
            df = price_cache.get_price_data(
                symbol, start_date=start_dt.isoformat(), end_date=end_dt.isoformat(), interval="1d"
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
            if classes[a] != classes[b]:
                continue  # only correlate within the same asset class
            full_corr, rolling_median, overlap = compute_pair_correlation(closes[a], closes[b])
            if full_corr is None:
                continue
            row = {
                "symbol_a": a, "symbol_b": b, "asset_class": classes[a],
                "full_corr": full_corr, "rolling_corr_median": rolling_median,
                "overlap_bars": overlap,
            }
            insert_correlation_row(conn, run_id, row)
            inserted_pairs.append({"run_id": run_id, **row})
            pairs_for_grouping.append(row)

    # Greedy clustering per asset class.
    groups = greedy_group_pairs(pairs_for_grouping)
    inserted_groups = []
    for gid, g in enumerate(groups, start=1):
        row = {
            "group_id": gid, "asset_class": g["asset_class"],
            "symbols": ",".join(g["symbols"]), "mean_intra_corr": g["mean_intra_corr"],
        }
        insert_group_row(conn, run_id, row)
        inserted_groups.append({"run_id": run_id, **row})

    conn.commit()
    csv_pairs = dump_csv("correlation_pairs", inserted_pairs, "correlation")
    csv_groups = dump_csv("suggested_groups", inserted_groups, "correlation")
    print(f"\nStage 2 (correlation) done: {len(inserted_pairs)} pairs, "
          f"{len(inserted_groups)} suggested groups. run_id={run_id}.")
    print(f"CSV: {csv_pairs}, {csv_groups}")
    for g in inserted_groups:
        print(f"  group {g['group_id']} [{g['asset_class']}]: {g['symbols']} "
              f"(mean_corr={g['mean_intra_corr']:.3f})")
    return run_id


# ── Subprocess runner shared by stages 3/4/5 ─────────────────────────────────

def run_backtest_subprocess(assets, interval="1d", backtest_period="6m",
                             no_prediction=False, model_path=None, pred_samples=100):
    """
    Invoke strategy/kairos_strategies.py as a subprocess and return the parsed
    JSON export (summary, strategy_rankings, shadow_performance).

    Shared by stage 3 (oracle, no_prediction=True), stage 4 (base model,
    no_prediction=False, model_path=None) and stage 5 (finetuned model,
    no_prediction=False, model_path=<checkpoint>). kairos_strategies.py
    already exposes a `--model` flag for a local finetuned checkpoint path,
    so stage 5 reuses it rather than inventing a new `--model_path` flag.
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

    print(f"  [subprocess] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
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


def run_stage_oracle(conn, assets, interval="1d", backtest_period="6m", pred_samples=100):
    run_id = start_run(conn, "oracle", interval, {
        "assets": assets, "backtest_period": backtest_period, "pred_samples": pred_samples,
    })
    payload = run_backtest_subprocess(
        assets, interval=interval, backtest_period=backtest_period,
        no_prediction=True, model_path=None, pred_samples=pred_samples,
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
    return run_id


def run_stage_model(conn, stage, assets, interval="1d", backtest_period="6m",
                     pred_samples=100, model_path=None):
    """
    Shared implementation for stage 4 ('base') and stage 5 ('finetuned').
    Not executed in this environment (needs GPU + downloaded/finetuned Kronos
    weights) but fully wired: parameterizes run_backtest_subprocess with
    no_prediction=False and (for 'finetuned') a --model checkpoint path.
    """
    assert stage in ("base", "finetuned")
    run_id = start_run(conn, stage, interval, {
        "assets": assets, "backtest_period": backtest_period,
        "pred_samples": pred_samples, "model_path": model_path,
    })
    payload = run_backtest_subprocess(
        assets, interval=interval, backtest_period=backtest_period,
        no_prediction=False, model_path=model_path, pred_samples=pred_samples,
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
             row["oracle_sharpe"], row["oracle_signals"], row["oracle_win_rate"], row["oracle_avg_pnl_per_trade"], row["oracle_run_id"],
             row["base_sharpe"], row["base_signals"], row["base_win_rate"], row["base_avg_pnl_per_trade"], row["base_run_id"], row["base_model_path"],
             row["signals_per_week"], int(row["viable"])),
        )
    conn.commit()

    # Write CSV
    rows_for_csv = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        rows_for_csv.append(row_dict)

    csv_path = dump_csv("viability_report", rows_for_csv, "auto")
    return csv_path


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


def main():
    parser = argparse.ArgumentParser(description="Kairos staged asset-discovery pipeline")
    parser.add_argument("--stage", required=True,
                         choices=["universe", "correlation", "oracle", "base", "finetuned"])
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--backtest_period", default="6m")
    parser.add_argument("--pred_samples", type=int, default=100)
    parser.add_argument("--assets", nargs="+", default=None, metavar="SYM")
    parser.add_argument("--group_id", type=int, default=None)
    parser.add_argument("--asset_class", default=None, choices=["crypto", "equity", "fx_commodity"],
                         help="Restrict --stage correlation to one asset class")
    # TODO: kairos_strategies.py has no dedicated --model_path flag; it reuses
    # --model for the local checkpoint path, which we forward as model_path here.
    parser.add_argument("--model_path", default=None,
                         help="Finetuned Kronos checkpoint path (stage=finetuned only)")
    args = parser.parse_args()

    conn = get_connection()

    if args.stage == "universe":
        run_stage_universe(conn, interval=args.interval)
    elif args.stage == "correlation":
        run_stage_correlation(conn, asset_class_filter=args.asset_class)
    elif args.stage == "oracle":
        assets = args.assets
        if args.group_id is not None:
            assets = _group_symbols_from_db(conn, args.group_id)
        if not assets:
            raise SystemExit("--stage oracle requires --assets SYM... or --group_id N")
        run_stage_oracle(conn, assets, interval=args.interval,
                          backtest_period=args.backtest_period, pred_samples=args.pred_samples)
    elif args.stage in ("base", "finetuned"):
        assets = args.assets
        if args.group_id is not None:
            assets = _group_symbols_from_db(conn, args.group_id)
        if not assets:
            raise SystemExit(f"--stage {args.stage} requires --assets SYM... or --group_id N")
        run_stage_model(conn, args.stage, assets, interval=args.interval,
                         backtest_period=args.backtest_period, pred_samples=args.pred_samples,
                         model_path=args.model_path if args.stage == "finetuned" else None)

    conn.close()


if __name__ == "__main__":
    main()
