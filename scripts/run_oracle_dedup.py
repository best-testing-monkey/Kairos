#!/usr/bin/env python3
"""Run --stage oracle (or naive) over a deduped subset of suggested_groups.

Runs `select_deduped_groups()` through `kairos_pipeline.run_stage_oracle()` /
`run_stage_naive()` across a process pool of workers instead of one group at
a time. Supersedes the original sequential scratchpad version of this script
(same name, used 2026-08-26): that version processed one group at a time via
a blocking `subprocess.run()` per group, which measured (via `ps`) at only
~4 cores' worth of BLAS work on average on a 16-logical-core box -- most
cores sat idle (the "one core pegged, others 5-40%" pattern was Linux's
scheduler load-balancing that one hot process across cores, not real
parallelism). Root cause turned out to be mostly single-threaded Python (the
day-by-day, strategy-by-strategy loop is GIL-bound, not BLAS-bound) --
confirmed via a controlled 4-vs-4-group benchmark: 4 parallel workers gave a
3.78x speedup over the sequential approach, close to linear. Each worker
subprocess is pinned to 1 BLAS thread so N workers approximate N busy cores
instead of oversubscribing.

`--stage naive` runs the same groups through the naive baseline (oracle's
real decision, re-anchored to a genuinely no-peek entry and resolved only
against later real bars) instead of oracle's perfect-foresight peek -- see
run_stage_naive's docstring for why its results deliberately don't feed the
disabled_strategies gate the way oracle's do.

Box has 8 physical / 16 logical cores; default --workers=8 keeps to physical
cores since SMT doesn't reliably double throughput for GIL-bound work like
this.

Safe to re-run/resume: already-inserted groups are skipped (same
assets/interval/backtest_period/stage exists-check as before).

Usage:
    uv run scripts/run_oracle_dedup.py <correlation_run_id> [--stage oracle|naive] [--workers N] [--limit N]
"""
import argparse
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "strategy"))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pipeline_results.db")

INTERVAL = "1d"
BACKTEST_PERIOD = "6m"
PRED_SAMPLES = 100
DISABLE_MIN_SIGNALS = 5

# ponytail: pin each subprocess to 1 BLAS thread so N workers ~= N busy cores
# instead of oversubscribing; upgrade to per-worker core pinning (taskset) if
# BLAS libraries ever ignore these env vars on this box.
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def _connect():
    import kairos_pipeline as kp
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(kp.SCHEMA)
    return conn


def _run_one(group_id, assets, stage):
    """Runs in a forked worker process. Must stay at module level (pickling)."""
    os.environ.update(_THREAD_ENV)
    import kairos_pipeline as kp
    conn = _connect()
    assets_key = ",".join(sorted(assets))
    try:
        exists = conn.execute(
            "SELECT run_id FROM oracle_results WHERE assets=? AND interval=? "
            "AND backtest_period=? AND stage=? LIMIT 1",
            (assets_key, INTERVAL, BACKTEST_PERIOD, stage),
        ).fetchone()
        if exists:
            return (group_id, assets_key, "skip", exists[0])
        if stage == "naive":
            run_id = kp.run_stage_naive(
                conn, assets, interval=INTERVAL, backtest_period=BACKTEST_PERIOD,
                pred_samples=PRED_SAMPLES,
            )
        else:
            run_id = kp.run_stage_oracle(
                conn, assets, interval=INTERVAL, backtest_period=BACKTEST_PERIOD,
                pred_samples=PRED_SAMPLES, disable_min_signals=DISABLE_MIN_SIGNALS,
            )
        return (group_id, assets_key, "done", run_id)
    except Exception as exc:
        return (group_id, assets_key, "fail", str(exc))
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("correlation_run_id", type=int)
    parser.add_argument("--stage", choices=["oracle", "naive"], default="oracle",
                         help="Which no-model mode to sweep (default: oracle)")
    parser.add_argument("--workers", type=int, default=8,
                         help="Parallel subprocess workers (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N groups (quick wiring test)")
    args = parser.parse_args()

    import kairos_pipeline as kp
    conn = _connect()
    groups = kp.select_deduped_groups(conn, args.correlation_run_id)
    conn.close()
    if args.limit:
        groups = groups[: args.limit]
    print(f"Total {args.stage} invocations planned: {len(groups)} across {args.workers} workers")

    done = skipped = failed = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, gid, assets, args.stage): (gid, assets) for gid, assets in groups}
        for i, fut in enumerate(as_completed(futures), 1):
            group_id, assets_key, status, info = fut.result()
            if status == "skip":
                skipped += 1
            elif status == "done":
                done += 1
            else:
                failed += 1
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta_min = (len(groups) - i) / rate / 60 if rate > 0 else float("inf")
            print(f"[{i}/{len(groups)}] [{status}] group {group_id} ({assets_key}) {info} "
                  f"eta_min={eta_min:.0f}")

    print(f"\nParallel {args.stage} sweep done: {done} run, {skipped} skipped, {failed} failed, "
          f"total={len(groups)}.")


if __name__ == "__main__":
    main()
