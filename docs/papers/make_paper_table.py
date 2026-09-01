#!/usr/bin/env python3
"""Rebuild paper_table.json (the input build_paper.py renders) from the database.

Same matched-sample rules the paper states in section 3.3/3.4 and that
audit_paper.py re-derives independently:
  * only rows produced under the current walk-forward exit rule (see EXIT_RULE_CUTOFF),
  * groups present in all three fresh sweeps, matched on (assets, interval, backtest_period),
  * group rows with fewer than 3 signals dropped,
  * the seven trend_following aliases dropped,
  * strategies that fired in all three regimes kept, ordered by base Sharpe.
"""
import json, pathlib, sqlite3

DB = "/media/baz/MonkeyWorks/PycharmProjects/Kairos/data/pipeline_results.db"
HERE = pathlib.Path(__file__).parent

# One exit rule for all three stages landed 2026-08-29 (CLAUDE.md, "One exit rule
# for oracle, base and naive"). Rows written before the re-sweep that followed were
# scored one bar ahead and force-closed, so they are not comparable and are excluded.
EXIT_RULE_CUTOFF = "2026-08-30"

ALIAS = {"cds_spread_filter", "cot_positioning_filter", "dark_pool_filter", "fractal_dimension",
         "gaussian_process", "insider_cluster", "onchain_flow_filter"}  # trend_following kept

STAGES = (("oracle_results", "oracle"), ("oracle_results", "naive"), ("model_results", "base"))

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row


def fresh(table, stage):
    return conn.execute(
        f"""SELECT o.strategy_name, o.sharpe, o.signal_count, o.win_rate, o.avg_pnl_per_trade,
                   o.assets, o.interval, o.backtest_period
            FROM {table} o JOIN runs r ON r.run_id = o.run_id
            WHERE o.stage = ? AND r.timestamp >= ?""", (stage, EXIT_RULE_CUTOFF)).fetchall()


R = {stage: fresh(t, stage) for t, stage in STAGES}
gkey = lambda r: (r["assets"], r["interval"], r["backtest_period"])
matched = set.intersection(*({gkey(r) for r in rows} for rows in R.values()))


def agg(rows):
    out = {}
    for r in rows:
        if r["signal_count"] < 3 or gkey(r) not in matched or r["strategy_name"] in ALIAS:
            continue
        d = out.setdefault(r["strategy_name"], {"n": 0, "sh": 0.0, "win": 0.0, "pnl": 0.0})
        n = r["signal_count"]
        d["n"] += n
        d["sh"] += r["sharpe"] * n
        d["win"] += r["win_rate"] * n
        d["pnl"] += r["avg_pnl_per_trade"] * n
    return {k: {"n": v["n"], "sh": v["sh"] / v["n"], "win": v["win"] / v["n"], "pnl": v["pnl"] / v["n"]}
            for k, v in out.items()}


S = {s: agg(rows) for s, rows in R.items()}
common = sorted(set.intersection(*(set(v) for v in S.values())), key=lambda k: -S["base"][k]["sh"])

# ---------------------------------------------------------------------------
# The naive regime was rebuilt on 2026-09-01 (CLAUDE.md, "Naive baseline"): it
# forecasts the withheld last bar instead of reusing the oracle's decision. The
# paper carries a "superseded" note about it, and that note needs measured
# numbers rather than adjectives. Computed against the pre-rebuild backup while
# it still exists; if the backup has been deleted, the previously recorded block
# is preserved rather than dropped, so re-running this never silently strips the
# figures out of the paper.
BACKUP = "/media/baz/MonkeyWorks/PycharmProjects/Kairos/data/pipeline_results.db.bak-20260901-prenaive-nopeek"


def naive_revision():
    import os, statistics as stx
    if not os.path.exists(BACKUP):
        try:
            return json.loads((HERE / "paper_table.json").read_text()).get("naive_revision")
        except Exception:
            return None
    ob = sqlite3.connect(BACKUP)
    cell = lambda cur: {(r[0], r[1]): r[2] for r in cur}
    q = """SELECT assets, strategy_name, sharpe FROM oracle_results
           WHERE stage='naive' AND interval='1d' AND backtest_period='6m'"""
    old_c, new_c = cell(ob.execute(q)), cell(conn.execute(q))
    ob.close()
    keys = [k for k in new_c if k in old_c]
    if not keys:
        return None
    deltas = [new_c[k] - old_c[k] for k in keys]
    return {
        "cells": len(keys),
        "identical": sum(1 for d in deltas if d == 0.0),
        "pct_identical": round(100.0 * sum(1 for d in deltas if d == 0.0) / len(keys), 1),
        "mean_delta": round(stx.mean(deltas), 4),
        "median_delta": round(stx.median(deltas), 4),
        "old_median": round(stx.median([old_c[k] for k in keys]), 3),
        "new_median": round(stx.median([new_c[k] for k in keys]), 3),
    }


data = {
    "matched_groups": len(matched),
    "naive_revision": naive_revision(),
    "rows": [{"name": k,
              "n": {s: S[s][k]["n"] for s in S},
              "sh": {s: round(S[s][k]["sh"], 3) for s in S},
              "win": {s: round(S[s][k]["win"], 4) for s in S},
              "pnl": {s: round(S[s][k]["pnl"], 6) for s in S}} for k in common],
}
(HERE / "paper_table.json").write_text(json.dumps(data, indent=1) + "\n")
print(f"wrote {HERE / 'paper_table.json'}: {len(matched)} matched groups, {len(common)} strategies")
for s in S:
    print(f"  {s:7s} {len(S[s]):3d} strategies, {sum(S[s][k]['n'] for k in common):,} signals in the common set")
