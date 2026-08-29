#!/usr/bin/env python3
"""Backfill strategy_class_stats from existing oracle_results/model_results rows.

New sweeps attribute each signal to its own symbol's class at source, which is
exact. Historical rows cannot be split that finely -- the per-symbol breakdown
was discarded before anything was persisted -- so this derives each row's class
from its GROUP's composition instead:

  - all symbols in the group share one class  -> that class (98% of groups)
  - the group spans two or more classes       -> 'mixed'

'mixed' rows are deliberately not attributed to any single class: they still
count toward corpus figures (which come from the original tables regardless),
but a per-class read will not see them. Re-running those groups through a sweep
is the only way to split them properly.

Idempotent (INSERT OR REPLACE on the natural key) and safe to re-run. No GPU.

Usage:
    uv run scripts/backfill_class_stats.py [--dry-run] [--db PATH]
"""
import argparse
import os
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "strategy"))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pipeline_results.db")

SOURCES = [
    ("oracle_results", ("oracle", "naive"), False),
    ("model_results", ("base", "finetuned"), True),
]


def group_class(assets: str, classify) -> str:
    """One class for a whole group, or 'mixed' when it spans several."""
    classes = {classify(s) for s in assets.split(",") if s.strip()}
    if not classes:
        return "mixed"
    return classes.pop() if len(classes) == 1 else "mixed"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written, change nothing")
    args = ap.parse_args()

    import kairos_pipeline as kp
    from kairos_backtest import asset_class_of_symbol

    conn = sqlite3.connect(args.db)
    conn.executescript(kp.SCHEMA)  # creates strategy_class_stats if absent
    conn.row_factory = sqlite3.Row

    totals = Counter()
    for table, stages, has_model_path in SOURCES:
        cols = ("run_id, stage, strategy_name, sharpe, signal_count, win_rate, "
                "avg_pnl_per_trade, assets, interval, backtest_period")
        if has_model_path:
            cols += ", model_path"
        for stage in stages:
            rows = conn.execute(
                f"SELECT {cols} FROM {table} WHERE stage=?", (stage,)).fetchall()
            for r in rows:
                cls = group_class(r["assets"] or "", asset_class_of_symbol)
                totals[cls] += 1
                totals[f"{stage}:{cls}"] += 1
                if args.dry_run:
                    continue
                kp.insert_class_stat_row(conn, r["run_id"], {
                    "stage": r["stage"],
                    "model_path": r["model_path"] if has_model_path else None,
                    "strategy_name": r["strategy_name"],
                    "asset_class": cls,
                    "sharpe": r["sharpe"],
                    "signal_count": r["signal_count"],
                    "win_rate": r["win_rate"],
                    "avg_pnl_per_trade": r["avg_pnl_per_trade"],
                    "assets": r["assets"],
                    "interval": r["interval"],
                    "backtest_period": r["backtest_period"],
                })
            print(f"  {table}/{stage}: {len(rows)} rows")
    if not args.dry_run:
        conn.commit()

    print("\nBy derived class:")
    for cls in ("equity", "crypto", "fx_commodity", "mixed"):
        if totals[cls]:
            print(f"  {cls:14s} {totals[cls]:7d}")
    mixed_pct = totals["mixed"] / max(sum(totals[c] for c in
                ("equity", "crypto", "fx_commodity", "mixed")), 1) * 100
    print(f"\n'mixed' (not attributable to one class): {mixed_pct:.1f}% of rows")
    if args.dry_run:
        print("\n(dry run -- nothing written)")
    else:
        n = conn.execute("SELECT COUNT(*) FROM strategy_class_stats").fetchone()[0]
        print(f"strategy_class_stats now holds {n} rows")
    conn.close()


if __name__ == "__main__":
    main()
