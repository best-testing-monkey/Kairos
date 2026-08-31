#!/usr/bin/env python3
"""Controlled per-group speed/VRAM benchmark across Kronos model variants.

Every earlier timing comparison in this project was confounded: the small and
mini sweeps ran with different worker counts, different pipelining settings, and
(worse) with orphaned pool workers from killed runs competing for the GPU. This
script exists to produce ONE number per model that can actually be quoted.

Controls, all deliberate:

  * Identical groups for every model (stratified 10x 1-asset + 10x 4-asset, since
    the paired corpus is bimodal and group size dominates cost).
  * Identical backtest config for every model -- interval, period and
    pred_samples are pinned as constants below and echoed into the output, so a
    future run can be checked for having moved the goalposts.
  * Serial execution, one backtest at a time. Parallelism is what made the
    previous numbers unquotable; it is deliberately absent here.
  * INTERLEAVED by group (base, small, mini on group 1, then group 2, ...)
    rather than all-of-one-model-then-the-next, so thermal drift or any slow
    background drift hits all three models equally instead of penalising
    whichever ran last.
  * No prediction cache. KAIROS_PRED_CACHE_DIR is explicitly cleared from the
    subprocess environment, so every run does its own real GPU work -- a warm
    cache would measure the cache, not the model.
  * Peak VRAM is sampled per run from this process's own child PID via
    nvidia-smi --query-compute-apps, not whole-GPU memory.used, which would
    include the display server.

The box must be otherwise idle. Check for orphaned pool workers first:
    ps -eo pid,ppid,cmd | grep -E 'run_model_parallel|kairos_strategies' | awk '$2==1'

Usage:
    uv run scripts/benchmark_models.py                      # 20 groups x 3 models
    uv run scripts/benchmark_models.py --groups 6 --models small,mini
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics as st
import subprocess
import sys
import tempfile
import threading
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "strategy"))
from kairos.models import resolve as resolve_model  # noqa: E402

DB_PATH = os.path.join(REPO_ROOT, "data", "pipeline_results.db")
STRATEGIES = os.path.join(REPO_ROOT, "strategy", "kairos_strategies.py")

# ---- PINNED BENCHMARK CONFIG (identical for every model; echoed into output) --
INTERVAL = "1d"
BACKTEST_PERIOD = "6m"
PRED_SAMPLES = 100
# ------------------------------------------------------------------------------


def select_groups(conn, n_total, stages):
    """Deterministic, stratified: half 1-asset, half 4-asset, covered by every stage."""
    sql = " INTERSECT ".join(
        f"SELECT DISTINCT assets FROM model_results WHERE stage='{s}' "
        f"AND interval=? AND backtest_period=?" for s in stages
    )
    params = []
    for _ in stages:
        params += [INTERVAL, BACKTEST_PERIOD]
    rows = [r[0] for r in conn.execute(sql, params).fetchall()]
    by_size: dict[int, list[str]] = {}
    for a in rows:
        by_size.setdefault(a.count(",") + 1, []).append(a)
    out = []
    half = n_total // 2
    for size in (1, 4):
        out += sorted(by_size.get(size, []))[:half]
    # Top up from anything else if a stratum was short.
    if len(out) < n_total:
        rest = sorted(a for a in rows if a not in out)
        out += rest[: n_total - len(out)]
    return out[:n_total]


def _peak_vram_of(pid_box, stop, out_box, interval=0.5):
    """Sample VRAM of our own child pid (and its descendants) until stop is set."""
    peak = 0.0
    while not stop.is_set():
        pid = pid_box.get("pid")
        if pid:
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=8, check=True,
                )
                total = 0.0
                for line in res.stdout.strip().splitlines():
                    if not line.strip():
                        continue
                    spid, mib = [x.strip() for x in line.split(",")[:2]]
                    # The uv wrapper forks; count any pid in our process group.
                    try:
                        if os.getpgid(int(spid)) == os.getpgid(pid):
                            total += float(mib)
                    except (ProcessLookupError, PermissionError, ValueError):
                        continue
                peak = max(peak, total)
            except Exception:
                pass
        time.sleep(interval)
    out_box["peak"] = peak


def run_one(assets, model_id):
    """Run one backtest serially; return (seconds, peak_vram_mib, ok)."""
    env = dict(os.environ)
    env.pop("KAIROS_PRED_CACHE_DIR", None)  # measure the model, not the cache
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_json = tf.name
    cmd = ["uv", "run", STRATEGIES,
           "--interval", INTERVAL, "--backtest_period", BACKTEST_PERIOD,
           "--pred_samples", str(PRED_SAMPLES),
           "--assets", *assets.split(","),
           "--export_json", out_json]
    if model_id:
        cmd += ["--model", model_id]

    pid_box, out_box, stop = {}, {}, threading.Event()
    sampler = threading.Thread(target=_peak_vram_of, args=(pid_box, stop, out_box), daemon=True)
    sampler.start()
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    pid_box["pid"] = proc.pid
    rc = proc.wait()
    elapsed = time.time() - t0
    stop.set()
    sampler.join(timeout=5)
    try:
        os.unlink(out_json)
    except OSError:
        pass
    return elapsed, out_box.get("peak", 0.0), rc == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--models", default="base,small,mini")
    ap.add_argument("--json", default=os.path.join(REPO_ROOT, "data", "model_benchmark.json"))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    conn = sqlite3.connect(DB_PATH)
    groups = select_groups(conn, args.groups, models)
    if len(groups) < args.groups:
        print(f"WARNING: only {len(groups)} groups covered by all of {models}")

    print(f"Benchmark: {len(groups)} groups x {len(models)} models, serial, interleaved.")
    print(f"Pinned config: interval={INTERVAL} period={BACKTEST_PERIOD} "
          f"pred_samples={PRED_SAMPLES}, prediction cache DISABLED.")
    print(f"Groups: {sum(1 for g in groups if ',' not in g)} single-asset, "
          f"{sum(1 for g in groups if ',' in g)} multi-asset\n")

    results = {m: [] for m in models}
    for i, assets in enumerate(groups, 1):
        for m in models:
            # base must run as the pipeline's default (model_path None), matching
            # how stage='base' rows are produced.
            model_id = None if m == "base" else resolve_model(m)["model_id"]
            secs, vram, ok = run_one(assets, model_id)
            results[m].append({"assets": assets, "n_assets": assets.count(",") + 1,
                               "seconds": secs, "peak_vram_mib": vram, "ok": ok})
            flag = "" if ok else "  *** FAILED ***"
            print(f"[{i}/{len(groups)}] {m:6s} {assets[:34]:34s} "
                  f"{secs:7.1f}s  {vram:7.0f} MiB{flag}", flush=True)

    print("\n" + "=" * 68)
    print(f"{'model':8s} {'median s':>9s} {'mean s':>8s} {'1-asset':>9s} "
          f"{'4-asset':>9s} {'peak VRAM':>10s} {'fails':>6s}")
    print("-" * 68)
    summary = {}
    for m in models:
        ok = [r for r in results[m] if r["ok"]]
        if not ok:
            print(f"{m:8s} {'ALL FAILED':>9s}")
            continue
        secs = [r["seconds"] for r in ok]
        s1 = [r["seconds"] for r in ok if r["n_assets"] == 1]
        s4 = [r["seconds"] for r in ok if r["n_assets"] > 1]
        vram = max(r["peak_vram_mib"] for r in ok)
        summary[m] = {"median_s": st.median(secs), "mean_s": st.mean(secs),
                      "median_1asset_s": st.median(s1) if s1 else None,
                      "median_4asset_s": st.median(s4) if s4 else None,
                      "peak_vram_mib": vram, "n_ok": len(ok),
                      "n_failed": len(results[m]) - len(ok)}
        d = summary[m]
        print(f"{m:8s} {d['median_s']:>9.1f} {d['mean_s']:>8.1f} "
              f"{(d['median_1asset_s'] or 0):>9.1f} {(d['median_4asset_s'] or 0):>9.1f} "
              f"{vram:>9.0f}M {d['n_failed']:>6d}")

    with open(args.json, "w") as fh:
        json.dump({"config": {"interval": INTERVAL, "backtest_period": BACKTEST_PERIOD,
                              "pred_samples": PRED_SAMPLES, "serial": True,
                              "cache_disabled": True, "interleaved": True},
                   "groups": groups, "summary": summary, "runs": results}, fh, indent=2)
    print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
