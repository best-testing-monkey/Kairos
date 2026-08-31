#!/usr/bin/env python3
"""Paired comparison of prediction regimes on an identical group set.

Compares model stages (base / small / mini) against each other and against the
naive floor and oracle ceiling, on the EXACT set of groups all requested stages
cover -- an inner join on `assets`, never a union. An unpaired comparison across
stages with different coverage is the failure mode this script exists to avoid
(see the 2026-08-29 coverage note in CLAUDE.md: the deduped list is 95% equity,
and stages do not cover the same groups by default).

Three methodological rules are enforced here, not left to the caller:

  1. Only rows on the CURRENT exit rule (runs.timestamp >= EXIT_RULE_CUTOFF).
     Older rows were scored one bar ahead and force-closed -- a different ruler.
  2. The seven trend_following aliases are collapsed. Each wraps
     TrendFollowingStrategy and gates on a context key the pipeline never
     supplies, so all eight are byte-identical: one behaviour would otherwise
     vote eight times and drag every median toward it.
  3. Sharpe is never recombined across groups by averaging a ratio -- per-strategy
     figures are medians across groups, which is what the papers report.

Usage:
    uv run scripts/compare_model_stages.py --stages base,small
    uv run scripts/compare_model_stages.py --stages base,small,mini --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics as st
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "data", "pipeline_results.db")

EXIT_RULE_CUTOFF = "2026-08-30"
INTERVAL = "1d"
BACKTEST_PERIOD = "6m"

# Collapse: all seven wrap TrendFollowingStrategy and degrade to a pass-through.
ALIAS = {"cds_spread_filter", "cot_positioning_filter", "dark_pool_filter",
         "fractal_dimension", "gaussian_process", "insider_cluster",
         "onchain_flow_filter"}  # trend_following itself is kept

# stage -> (table, extra predicate). oracle/naive live in oracle_results.
_TABLES = {
    "oracle": ("oracle_results", "stage='oracle'"),
    "naive": ("oracle_results", "stage='naive'"),
}


def _table_for(stage: str) -> tuple[str, str]:
    return _TABLES.get(stage, ("model_results", f"stage='{stage}'"))


def load_stage(conn, stage):
    """{assets: {strategy: (sharpe, signal_count, win_rate)}} on the current rule."""
    table, pred = _table_for(stage)
    rows = conn.execute(
        f"SELECT m.assets, m.strategy_name, m.sharpe, m.signal_count, m.win_rate "
        f"FROM {table} m JOIN runs r ON r.run_id = m.run_id "
        f"WHERE m.{pred} AND m.interval=? AND m.backtest_period=? "
        f"  AND r.timestamp >= ? "
        f"ORDER BY m.run_id",
        (INTERVAL, BACKTEST_PERIOD, EXIT_RULE_CUTOFF),
    ).fetchall()
    out: dict[str, dict[str, tuple]] = {}
    for assets, strat, sharpe, n, win in rows:
        if strat in ALIAS:
            continue
        if sharpe is None:
            continue
        # Latest run wins if a group was swept more than once. The ORDER BY
        # is load-bearing, not cosmetic: 6 mini groups were swept twice on
        # 2026-08-31 (orphaned pool workers from a killed sweep kept running
        # alongside its replacement), and without ordering the overwrite
        # picked a row at the DB's whim, making the comparison unreproducible.
        out.setdefault(assets, {})[strat] = (sharpe, n or 0, win)
    return out


def median_or_none(xs):
    return st.median(xs) if xs else None


# Below this, a per-class cell is too thin to read (mirrors
# kairos_pipeline.CLASS_STATS_MIN_SIGNALS, which Baz set for statistical
# relevance -- not a placeholder).
CLASS_MIN_SIGNALS = 30


def load_stage_by_class(conn, stage):
    """{(assets, asset_class): {strategy: (sharpe, signal_count)}}.

    Reads strategy_class_stats directly rather than via
    kairos_pipeline.strategy_class_stats(), whose corpus-table selector is
    `stage in ("base","finetuned")` -- it would silently read oracle_results
    for stage='small'/'mini' and return nothing.

    Per-class Sharpe is exact within a class (computed from that class's own
    pooled pnl_list). It must never be recombined across classes into a
    corpus figure -- Sharpe is a ratio and does not recombine.
    """
    rows = conn.execute(
        "SELECT c.assets, c.asset_class, c.strategy_name, c.sharpe, c.signal_count "
        "FROM strategy_class_stats c JOIN runs r ON r.run_id = c.run_id "
        "WHERE c.stage=? AND c.interval=? AND c.backtest_period=? "
        "  AND r.timestamp >= ? "
        "ORDER BY c.run_id",
        (stage, INTERVAL, BACKTEST_PERIOD, EXIT_RULE_CUTOFF),
    ).fetchall()
    out: dict[tuple, dict] = {}
    for assets, cls, strat, sharpe, n in rows:
        if strat in ALIAS or sharpe is None:
            continue
        # 'mixed' marks backfilled rows whose per-symbol split was already lost.
        if cls == "mixed":
            continue
        out.setdefault((assets, cls), {})[strat] = (sharpe, n or 0)
    return out


def report_by_class(conn, stages, context):
    """Per-(strategy, asset class) comparison, paired per (group, class)."""
    data = {s: load_stage_by_class(conn, s) for s in stages + context}

    classes = sorted({cls for d in data.values() for (_a, cls) in d})
    print("=" * 74)
    print("BY ASSET CLASS  (paired per (group, class); Sharpe never pooled across classes)")
    print("=" * 74)

    for cls in classes:
        keys = {k for k in data[stages[0]] if k[1] == cls}
        for s in stages[1:]:
            keys &= {k for k in data[s] if k[1] == cls}
        if not keys:
            print(f"\n[{cls}] no groups covered by all compared stages -- skipped\n")
            continue

        print(f"\n[{cls}]  paired groups: {len(keys)}")
        if len(keys) < 20:
            print(f"  *** UNDERPOWERED: {len(keys)} groups. Treat as indicative only. ***")
        print(f"  {'stage':10s} {'med Sharpe':>11s} {'mean':>9s} {'>0':>7s} "
              f"{'signals':>10s} {'thin cells':>11s}")
        print("  " + "-" * 62)
        for s in stages + context:
            sh, sigs, pos, thin = [], 0, 0, 0
            for k in keys:
                for _strat, (v, n) in data[s].get(k, {}).items():
                    if n < CLASS_MIN_SIGNALS:
                        thin += 1
                        continue
                    sh.append(v)
                    sigs += n
                    if v > 0:
                        pos += 1
            if not sh:
                print(f"  {s:10s} {'--':>11s} {'--':>9s} {'--':>7s} "
                      f"{sigs:>10,d} {thin:>11d}")
                continue
            print(f"  {s:10s} {st.median(sh):>11.3f} {st.mean(sh):>9.3f} "
                  f"{pos/len(sh)*100:>6.1f}% {sigs:>10,d} {thin:>11d}")

        # Strategies that flip sign between the two compared stages in this class.
        a, b = stages[0], stages[1]
        flips = []
        per = {}
        for k in keys:
            for strat in set(data[a].get(k, {})) & set(data[b].get(k, {})):
                va, na = data[a][k][strat]
                vb, nb = data[b][k][strat]
                if na < CLASS_MIN_SIGNALS or nb < CLASS_MIN_SIGNALS:
                    continue
                per.setdefault(strat, {"a": [], "b": []})
                per[strat]["a"].append(va)
                per[strat]["b"].append(vb)
        for strat, v in per.items():
            ma, mb = median_or_none(v["a"]), median_or_none(v["b"])
            if ma is None or mb is None or len(v["a"]) < 3:
                continue
            if (ma > 0) != (mb > 0):
                flips.append((strat, ma, mb, len(v["a"])))
        if flips:
            print(f"\n  sign flips between {a} and {b} in {cls} (n>=3 groups):")
            for strat, ma, mb, n in sorted(flips, key=lambda t: -abs(t[2] - t[1]))[:6]:
                print(f"    {strat:30s} {a}={ma:+.2f}  {b}={mb:+.2f}  (n={n})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="base,small",
                    help="Comma-separated stages to compare (default: base,small). "
                         "oracle/naive are read from oracle_results automatically.")
    ap.add_argument("--context", default="naive,oracle",
                    help="Extra stages to report as floor/ceiling context, only on "
                         "the paired group set (default: naive,oracle). Empty to skip.")
    ap.add_argument("--by-class", action="store_true",
                    help="Also segment by asset class (equity/crypto/fx_commodity). "
                         "Strategy quality is strongly class-dependent, so the pooled "
                         "figures above average away the effect you probably care about.")
    ap.add_argument("--json", default=None, help="Write the full result as JSON here.")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    context = [s.strip() for s in args.context.split(",") if s.strip()]
    if len(stages) < 2:
        sys.exit("need at least two --stages to compare")

    conn = sqlite3.connect(args.db)
    data = {s: load_stage(conn, s) for s in stages + context}

    # Pair on groups covered by every compared stage (context stages don't gate).
    paired = set(data[stages[0]])
    for s in stages[1:]:
        paired &= set(data[s])
    paired = sorted(paired)
    if not paired:
        sys.exit("no groups are covered by all requested stages")

    print(f"Paired groups: {len(paired)}  (stages compared: {', '.join(stages)})")
    for s in stages + context:
        print(f"  {s:8s} covers {len(data[s]):4d} groups total, "
              f"{len(set(data[s]) & set(paired)):4d} of the paired set")
    print()

    # ---- corpus medians over the paired set -------------------------------
    print(f"{'stage':10s} {'med Sharpe':>11s} {'mean Sharpe':>12s} "
          f"{'>0':>6s} {'strats':>7s} {'signals':>10s}")
    print("-" * 60)
    summary = {}
    for s in stages + context:
        sharpes, sigs, pos, strat_names = [], 0, 0, set()
        for g in paired:
            for strat, (sh, n, _w) in data[s].get(g, {}).items():
                sharpes.append(sh)
                sigs += n
                strat_names.add(strat)
                if sh > 0:
                    pos += 1
        summary[s] = {
            "median_sharpe": median_or_none(sharpes),
            "mean_sharpe": st.mean(sharpes) if sharpes else None,
            "pct_positive": (pos / len(sharpes) * 100) if sharpes else None,
            "n_strategies": len(strat_names),
            "n_signals": sigs,
            "n_observations": len(sharpes),
        }
        d = summary[s]
        print(f"{s:10s} {d['median_sharpe']:>11.3f} {d['mean_sharpe']:>12.3f} "
              f"{d['pct_positive']:>5.1f}% {d['n_strategies']:>7d} {d['n_signals']:>10,d}")
    print()

    # ---- per-strategy head-to-head between the two primary stages ---------
    a, b = stages[0], stages[1]
    per_strat = {}
    for g in paired:
        for strat in set(data[a].get(g, {})) & set(data[b].get(g, {})):
            per_strat.setdefault(strat, {"a": [], "b": []})
            per_strat[strat]["a"].append(data[a][g][strat][0])
            per_strat[strat]["b"].append(data[b][g][strat][0])

    deltas = []
    for strat, v in per_strat.items():
        ma, mb = median_or_none(v["a"]), median_or_none(v["b"])
        if ma is None or mb is None:
            continue
        deltas.append((mb - ma, strat, ma, mb, len(v["a"])))
    deltas.sort()

    print(f"Per-strategy median Sharpe, {b} minus {a}  (paired groups only)")
    print(f"{'strategy':32s} {a:>9s} {b:>9s} {'delta':>9s} {'n':>5s}")
    print("-" * 70)
    for d, strat, ma, mb, n in deltas[:8]:
        print(f"{strat:32s} {ma:>9.2f} {mb:>9.2f} {d:>+9.2f} {n:>5d}")
    if len(deltas) > 16:
        print(f"{'...':32s} {'':>9s} {'':>9s} {'':>9s}")
    for d, strat, ma, mb, n in deltas[-8:]:
        print(f"{strat:32s} {ma:>9.2f} {mb:>9.2f} {d:>+9.2f} {n:>5d}")
    print()

    better = sum(1 for d, *_ in deltas if d > 0)
    print(f"{b} beats {a} on {better}/{len(deltas)} strategies "
          f"({better/len(deltas)*100:.0f}%), median delta "
          f"{median_or_none([d for d, *_ in deltas]):+.3f}")

    # Spearman rank correlation without scipy: rank, then Pearson on ranks.
    if len(deltas) > 2:
        xs = [ma for _, _, ma, _, _ in deltas]
        ys = [mb for _, _, _, mb, _ in deltas]

        def rank(vs):
            order = sorted(range(len(vs)), key=lambda i: vs[i])
            r = [0.0] * len(vs)
            for pos, i in enumerate(order):
                r[i] = float(pos)
            return r

        rx, ry = rank(xs), rank(ys)
        mx, my = st.mean(rx), st.mean(ry)
        num = sum((p - mx) * (q - my) for p, q in zip(rx, ry))
        den = (sum((p - mx) ** 2 for p in rx) * sum((q - my) ** 2 for q in ry)) ** 0.5
        rho = num / den if den else float("nan")
        print(f"Rank correlation of per-strategy median Sharpe ({a} vs {b}): rho={rho:+.3f}")
        summary["rank_correlation"] = rho

    if args.by_class:
        print()
        report_by_class(conn, stages, context)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"paired_groups": len(paired), "stages": stages,
                       "summary": summary,
                       "per_strategy": [
                           {"strategy": s, f"{a}_median": ma, f"{b}_median": mb,
                            "delta": d, "n_groups": n}
                           for d, s, ma, mb, n in deltas]}, fh, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
