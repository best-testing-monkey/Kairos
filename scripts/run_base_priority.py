#!/usr/bin/env python3
"""Run --stage base over deduped correlation groups, prioritized by best oracle Sharpe, time-boxed.

Background: kairos_pipeline.py's correlation stage (greedy_group_pairs) emits
massively overlapping candidate groups at scraped-universe scale -- a single
popular symbol can appear in dozens of near-duplicate group combinations (see
docs/handoff for the 2026-08-26 IBKR-universe expansion session that surfaced
this: 13,472 raw groups reduced to 961 via kairos_pipeline.select_deduped_groups()).
Running the GPU-expensive `base` stage once per raw group would take days for no
extra coverage; this script also skips work already done and lets you spend a
bounded GPU budget on the most promising groups first.

What it does, each invocation:
  1. Builds the deduped group list via kairos_pipeline.select_deduped_groups()
     (greedy set-cover: minimal group set that still covers every survivor
     symbol at least once).
  2. Drops any group already present in model_results for stage='base' with
     this model_path -- so re-running this script is always safe/incremental,
     never redoes work.
  3. Ranks the remaining groups by MAX(sharpe) from oracle_results for that
     group's asset set (best-strategy-so-far, descending). Groups with no
     oracle result yet sort last, not excluded -- oracle_results is read fresh
     each invocation, so a group unranked this run may be ranked (and picked
     first) the next time you run this script, once oracle has caught up.
  4. Runs run_stage_model(..., stage="base") group by group until either the
     ranked list is exhausted or `max_hours` elapses. The time check only gates
     *starting* a new group -- an in-progress group is never killed mid-run.
     Whatever's left over is picked up cleanly by the next invocation.

Usage:
    uv run scripts/run_base_priority.py [max_hours]   # default 8.0
"""
from __future__ import annotations

import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "strategy"))
import kairos_pipeline as kp  # noqa: E402

INTERVAL = "1d"
BACKTEST_PERIOD = "6m"
PRED_SAMPLES = 100
STAGE = "base"
MODEL_PATH = None  # base uses the default Kronos-base model, not a finetuned checkpoint


def latest_correlation_run_id(conn, interval: str) -> int:
    row = conn.execute(
        "SELECT MAX(run_id) FROM runs WHERE stage='correlation' AND interval=?", (interval,)
    ).fetchone()
    if not row or row[0] is None:
        raise SystemExit(f"No correlation run found for interval={interval!r}. Run --stage correlation first.")
    return row[0]


def best_oracle_sharpe(conn, assets_key: str) -> float | None:
    """MAX sharpe across strategies for this exact group's asset set, or None if oracle hasn't run it yet."""
    row = conn.execute(
        "SELECT MAX(sharpe) FROM oracle_results WHERE assets=? AND interval=? AND backtest_period=?",
        (assets_key, INTERVAL, BACKTEST_PERIOD),
    ).fetchone()
    return row[0] if row else None


def main():
    max_hours = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    deadline = time.time() + max_hours * 3600

    conn = kp.get_connection(kp.DB_PATH)
    correlation_run_id = latest_correlation_run_id(conn, INTERVAL)
    groups = kp.select_deduped_groups(conn, correlation_run_id)

    # Filter to unprocessed, then rank by oracle Sharpe. Done as two passes (not
    # one sort with a DB lookup per compare) so the oracle query only runs once
    # per candidate group, not O(n log n) times during sorting.
    prioritized = []
    for group_id, assets in groups:
        assets_key = ",".join(sorted(assets))
        already_done = conn.execute(
            "SELECT run_id FROM model_results WHERE assets=? AND interval=? "
            "AND backtest_period=? AND stage=? AND model_path IS ? LIMIT 1",
            (assets_key, INTERVAL, BACKTEST_PERIOD, STAGE, MODEL_PATH),
        ).fetchone()
        if already_done:
            continue
        sharpe = best_oracle_sharpe(conn, assets_key)
        prioritized.append((group_id, assets, assets_key, sharpe))

    # None (unranked) sorts last; among ranked groups, highest Sharpe first.
    prioritized.sort(key=lambda g: (g[3] is None, -(g[3] or 0.0)))
    print(f"Unprocessed groups: {len(prioritized)} (of {len(groups)} total deduped, "
          f"correlation run_id={correlation_run_id}). Budget: {max_hours}h.")

    done = failed = 0
    t0 = time.time()
    for idx, (group_id, assets, assets_key, sharpe) in enumerate(prioritized, 1):
        if time.time() >= deadline:
            print(f"\n[budget] {max_hours}h elapsed, stopping before group {group_id}. "
                  f"{len(prioritized) - idx + 1} groups remain for next invocation.")
            break
        try:
            run_id = kp.run_stage_model(
                conn, STAGE, assets, interval=INTERVAL, backtest_period=BACKTEST_PERIOD,
                pred_samples=PRED_SAMPLES, model_path=MODEL_PATH,
            )
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta_min = (len(prioritized) - idx) / rate / 60 if rate > 0 else float("inf")
            print(f"[{idx}/{len(prioritized)}] [done] group {group_id} ({assets_key}) "
                  f"oracle_sharpe={sharpe} run_id={run_id} eta_min_full_remaining={eta_min:.0f}")
        except Exception as exc:
            failed += 1
            print(f"[{idx}/{len(prioritized)}] [FAIL] group {group_id} ({assets_key}): {exc}")

    print(f"\nBase priority sweep chunk done: {done} run, {failed} failed this invocation.")


if __name__ == "__main__":
    main()
