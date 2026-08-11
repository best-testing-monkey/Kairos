#!/usr/bin/env python3
"""Grid-search kairos_signal_replay's allocation/selection knobs for the most profitable config.

Sweeps top_k x max_pos_pct x a list of --signal-selection rule strings against an
already-precomputed replay window (run kairos_signal_replay.py --precompute first),
ranks by total_profit_eur, and prints the top N. Imports replay() directly instead of
shelling out per combo, since the precomputed closures in the db don't change between runs.

Overfitting warning (see CLAUDE.md "Offline Signal Replay"): a config that wins here
wins on THIS window only. Run the sweep over two non-overlapping windows and only trust
a config that ranks well on both, then validate the winner with a real kairos_papertrade.py
run before trusting it with capital -- replay's flat-fee cost model diverges from phantom's
live one.

Usage:
    uv run ./strategy/kairos_replay_sweep.py --interval 1d --start 2026-08-01 --end 2026-08-07 \\
        --capital 200 --top-k-grid 3,5,8,12 --max-pos-pct-grid 10,15,20,25 \\
        --selection-file selection_rules.txt
"""
import argparse
import itertools
import sqlite3
import sys
from pathlib import Path

from allocation import AllocationConfig
from kairos_signal_replay import _ensure_configured_db, replay, SIGNALS_DB_PATH
from signal_selection import parse_signal_selection, SignalSelectionError


def load_selection_rules(path: str | None) -> list[str | None]:
    """Rule strings to sweep; None always included first as the no-override baseline."""
    rules: list[str | None] = [None]
    if path:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                rules.append(line)
    return rules


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=SIGNALS_DB_PATH)
    p.add_argument("--interval", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--capital", type=float, required=True)
    p.add_argument("--top-k-grid", default="3,5,8,12")
    p.add_argument("--max-pos-pct-grid", default="10,15,20,25")
    p.add_argument("--selection-file", default=None,
                    help="File of --signal-selection rule strings, one per line. "
                         "The no-override default is always swept too.")
    p.add_argument("--min-trades", type=int, default=10,
                    help="Discard configs with fewer trades than this (avoid noise-driven winners)")
    p.add_argument("--top-n", type=int, default=10)
    args = p.parse_args(argv)

    top_ks = [int(x) for x in args.top_k_grid.split(",")]
    max_pos_pcts = [float(x) for x in args.max_pos_pct_grid.split(",")]
    rules = load_selection_rules(args.selection_file)

    _ensure_configured_db(args.db)
    conn = sqlite3.connect(args.db)

    results = []
    combos = list(itertools.product(top_ks, max_pos_pcts, rules))
    try:
        for i, (top_k, max_pos_pct, rule_str) in enumerate(combos, 1):
            selection_rule = None
            if rule_str is not None:
                try:
                    selection_rule = parse_signal_selection(rule_str)
                except SignalSelectionError as e:
                    print(f"[sweep] skip invalid rule {rule_str!r}: {e}", file=sys.stderr)
                    continue
            alloc_config = AllocationConfig(
                top_k=top_k, max_pos_pct=max_pos_pct, max_leverage=1.0,
                selection_rule=selection_rule,
            )
            metrics = replay(conn, args.interval, args.start, args.end, alloc_config, args.capital)
            metrics.update(top_k=top_k, max_pos_pct=max_pos_pct, selection=rule_str or "<default>")
            results.append(metrics)
            print(f"[sweep] {i}/{len(combos)} top_k={top_k} max_pos_pct={max_pos_pct} "
                  f"rule={rule_str or '<default>'} -> {metrics['total_profit_eur']:.2f} EUR "
                  f"({metrics['pct_profit']:.2f}%, {metrics['num_trades']} trades)", file=sys.stderr)
    finally:
        conn.close()

    ranked = [r for r in results if r["num_trades"] >= args.min_trades]
    ranked.sort(key=lambda r: r["total_profit_eur"], reverse=True)

    print(f"\n=== Top {min(args.top_n, len(ranked))} by total_profit_eur "
          f"(min {args.min_trades} trades, {len(results)} configs tried) ===")
    for r in ranked[:args.top_n]:
        print(f"{r['total_profit_eur']:>10.2f} EUR ({r['pct_profit']:>6.2f}%)  "
              f"top_k={r['top_k']:<3} max_pos_pct={r['max_pos_pct']:<5} "
              f"trades={r['num_trades']:<4} dd={r['pct_max_drawdown']:.2f}%  "
              f"selection={r['selection']}")

    if not ranked:
        print("No config met --min-trades threshold; lower --min-trades or widen the window.")


if __name__ == "__main__":
    main()
