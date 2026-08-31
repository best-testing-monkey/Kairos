#!/usr/bin/env python3
"""Two-phase parallel sweep driver for GPU-bound model stages (base/small/mini).

Measured fact: GPU utilization during a `--stage base`/`small`/`mini` sweep
averages only 28.8% -- ~71% of wall clock is non-GPU work (data fetch, the
GIL-bound per-day/per-strategy backtest loop inside kairos_strategies.py's
subprocess). This script serializes the GPU work and parallelizes everything
else, per chunk of groups:

  Phase 1 (GPU, serial by default -- --gpu-workers): populate the shared
    kairos_predcache disk cache for every (group, date) in the chunk, by
    replicating the exact enumeration + KairosSettings state
    KairosOrchestrator.run_backtest / kairos_strategies.py's __main__ block
    use, so the cache key this builds is byte-identical to what the real
    backtest subprocess will look up (see enumerate_backtest_dates() and
    _prewarm_group() below -- getting this wrong is the main failure mode).

  Phase 2 (CPU, --workers via ProcessPoolExecutor): run
    kairos_pipeline.run_stage_model() per group. That function shells out to
    `uv run strategy/kairos_strategies.py` per group; extra_env carries
    KAIROS_PRED_CACHE_DIR (+ max-bytes) into that subprocess so every
    prediction is a shared-cache hit and no worker ever loads a Kronos model.

Chunking (--chunk-size, default 16) is REQUIRED, not cosmetic: kairos_predcache
enforces a disk budget (default 2GiB, KAIROS_PRED_CACHE_MAX_BYTES) with
oldest-mtime eviction. Prewarming everything up front could evict a chunk's
own early entries before phase 2 reads them, silently degrading to per-worker
model loads. --cache-max-gb (default 8) sizes the budget generously relative
to one chunk's working set.

The group-selection/prioritization logic (unprocessed-for-stage, optional
--require-stage/--require-since freshness join, oracle-Sharpe-descending
ordering) is copied from scripts/run_base_priority.py's main() -- same
behavior, factored into select_prioritized_groups() here. Keep the two in
sync if this logic ever changes in either file.

Usage:
    uv run scripts/run_model_parallel.py [max_hours] --model small --stage small \\
        [--limit N] [--require-stage base] [--require-since YYYY-MM-DD] \\
        [--workers 8] [--gpu-workers 1|auto] [--gpu-workers-max 4] \\
        [--chunk-size 16] [--cache-max-gb 8] [--cache-dir PATH] [--pipeline]

--pipeline overlaps chunk N+1's phase 1 (GPU prewarm) with chunk N's phase 2
(CPU replay) -- measured near-symmetric phases with the GPU idle throughout
phase 2, so this cuts per-chunk wall clock toward max(phase1, phase2) instead
of phase1+phase2. Off by default; behaviour/output of the non-pipelined path
is unchanged. NOTE: two chunks' predictions are live in the shared predcache
at once under --pipeline (this chunk's phase 2 reads + the next chunk's phase
1 writes), roughly doubling the working set that --cache-max-gb must cover --
if a chunk's prewarm gets evicted (oldest-mtime) before its own phase 2 reads
it, that phase 2 silently degrades to per-worker model loads instead of
cache hits.

--control-file lets --workers/--gpu-workers/--chunk-size be rebalanced while
the sweep is running, without killing it: the file is re-read before each
chunk and any recognized keys override the current values from that chunk
onward (gpu_workers may also be the string "auto"). ponytail: chunk_size is
only honored live in the non-pipelined path -- --pipeline's lookahead
prefetch needs the chunk list built one chunk ahead of time, so a chunk_size
edit there takes effect for chunks not yet prefetched, one chunk later than
the sequential path. Example:
    echo '{"workers":2,"gpu_workers":4}' > data/sweep_control.json
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "strategy"))
import kairos_pipeline as kp  # noqa: E402
import kairos_predcache as kp_predcache  # noqa: E402  (reuse _read_mem_available_bytes, not a 2nd copy)
from kairos.models import resolve as resolve_model  # noqa: E402

INTERVAL = "1d"
BACKTEST_PERIOD = "6m"
PRED_SAMPLES = 100

DEFAULT_CACHE_DIR = os.path.join(REPO_ROOT, "data", "predcache_sweep")
DEFAULT_CONTROL_FILE = os.path.join(REPO_ROOT, "data", "sweep_control.json")
_PREWARM_GC_INTERVAL = 500  # mirrors kairos_papertrade.py's _PREWARM_GC_INTERVAL

# --gpu-workers auto sizing (see decide_gpu_workers() below / --gpu-workers-max
# help). per-worker estimates are seeds -- _phase1_with_vram_calibration()
# refines them chunk-to-chunk from what was actually observed.
VRAM_HEADROOM_MIB = 512
RAM_HEADROOM_MIB = 1024
# Seed for the FIXED --gpu-workers <int> path only (per_worker_vram_mib, fed
# to decide_gpu_workers()/_phase1_with_vram_calibration()) -- --gpu-workers
# auto no longer sizes off a single scalar at all, see
# estimate_group_vram_mib()/pack_waves() below. Raised 2026-08-31 from 3600 to
# 5200: base's actual measured 4-asset-group peak (scripts/benchmark_models.py,
# data/model_benchmark.json) is 5124 MiB, and 3600 was BELOW that -- an
# under-estimate on exactly the case this seed exists to protect chunk 1
# against, before any calibration data exists. An OOM here can deadlock the
# pool rather than fail cleanly (2026-08-31 incident) -- estimate high.
DEFAULT_PER_WORKER_VRAM_MIB = 5200.0
DEFAULT_PER_WORKER_RSS_MIB = 1100.0


# ── Per-group VRAM estimate table + wave packing (added 2026-08-31) ────────
# Measured via scripts/benchmark_models.py (serial, pred_samples=100, 1d/6m,
# peak per-process VRAM via nvidia-smi --query-compute-apps, 20 stratified
# groups x 3 models, 0 failures) -- see data/model_benchmark.json for the
# full per-group run data these two points per model were read off. Only
# n=1 and n=4 were actually measured; estimate_group_vram_mib() interpolates/
# extrapolates the rest.
_VRAM_TABLE_MIB = {
    "NeoQuasar/Kronos-base": {1: 1348.0, 4: 5124.0},
    "NeoQuasar/Kronos-small": {1: 682.0, 4: 3164.0},
    "NeoQuasar/Kronos-mini": {1: 562.0, 4: 2818.0},
}
_VRAM_TABLE_DEFAULT_MODEL = "NeoQuasar/Kronos-base"  # finetuned checkpoints are base-derived


def estimate_group_vram_mib(model_id, n_assets):
    """Estimate peak per-process VRAM (MiB) for one phase-1 prewarm worker
    running a group of `n_assets` assets under `model_id`. Linear
    interpolation between the measured n=1/n=4 points, linear extrapolation
    beyond n=4, never below the n=1 value. An unknown model_id (e.g. a local
    finetuned checkpoint path, which won't be a key in the table) falls back
    to base's profile -- the conservative choice, since finetuned checkpoints
    are Kronos-base derived."""
    row = _VRAM_TABLE_MIB.get(model_id) or _VRAM_TABLE_MIB[_VRAM_TABLE_DEFAULT_MODEL]
    v1, v4 = row[1], row[4]
    slope = (v4 - v1) / 3.0  # per additional asset from n=1 to n=4
    return max(v1 + slope * (n_assets - 1), v1)


def calibrate_vram_table(model_id, n_assets, observed_peak_mib):
    """Revise _VRAM_TABLE_MIB[model_id][n_assets] upward from an actually-
    observed per-process VRAM peak (see run_phase1_packed()'s per-wave
    sampling). Only ever raises the stored value -- a single low reading may
    just mean that worker hadn't peaked yet when sampled, not that the true
    peak fell. Downward revision across runs is out of scope: this table is
    a module-level dict, never written to disk, so there is no cross-run
    state to half-implement eviction for.

    ponytail: only the n=1/n=4 anchor points are ever calibrated (matching
    estimate_group_vram_mib()'s 2-point interpolation) -- an observation at
    some other group size (n=2/3, n>4) is not folded back in, since that
    would mean inventing a 3rd knot and a piecewise-interpolation scheme the
    bimodal 1-vs-4-asset corpus (see module docstring) doesn't need today.
    Add a real per-n table if a wider size distribution ever makes that
    worth it."""
    if n_assets not in (1, 4):
        return
    row = _VRAM_TABLE_MIB.setdefault(model_id, dict(_VRAM_TABLE_MIB[_VRAM_TABLE_DEFAULT_MODEL]))
    row[n_assets] = max(row.get(n_assets, 0.0), observed_peak_mib)


def pack_waves(groups, model_id, budget_mib, max_concurrent):
    """First-fit-decreasing bin packing of `groups` (list of (group_id,
    assets, assets_key, sharpe) -- the same tuple shape run_phase1/run_phase2
    chunks already use) into waves of VRAM-fitting, concurrently-runnable
    groups: sort by estimated VRAM descending, place each group into the
    first wave it fits (current wave total + its cost <= budget_mib, and the
    wave has fewer than max_concurrent members), else start a new wave.

    A group whose estimate alone exceeds budget_mib still gets its own wave
    of size 1 -- never dropped -- logged loudly, since the card likely can't
    actually hold it and an OOM is the expected outcome.

    Pure: no I/O, no GPU probing. Returns list[list[group]]."""
    sized = sorted(
        ((estimate_group_vram_mib(model_id, len(g[1])), g) for g in groups),
        key=lambda t: t[0], reverse=True,
    )
    waves = []  # list of [running_total_mib, [group, ...]]
    for cost, g in sized:
        if cost > budget_mib:
            print(f"[pack_waves] WARNING: group {g[2]!r} estimated {cost:.0f} MiB alone exceeds "
                  f"the {budget_mib:.0f} MiB budget -- giving it its own wave; OOM is likely.")
            waves.append([cost, [g]])
            continue
        for w in waves:
            if len(w[1]) < max_concurrent and w[0] + cost <= budget_mib:
                w[0] += cost
                w[1].append(g)
                break
        else:
            waves.append([cost, [g]])
    result = [w[1] for w in waves]
    assert sum(len(w) for w in result) == len(groups), "pack_waves lost or duplicated a group"
    return result

# ponytail: pin each subprocess to 1 BLAS thread so N workers ~= N busy cores
# instead of oversubscribing -- copied from scripts/run_oracle_dedup.py's
# _THREAD_ENV (same rationale: the backtest loop is GIL-bound Python, not
# BLAS-bound, so this just avoids oversubscription on the BLAS calls that do
# happen).
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


# ── Group selection (copied from scripts/run_base_priority.py's main() -- ──
# ── same filtering/priority logic; keep the two files in sync). ────────────

def latest_correlation_run_id(conn, interval: str) -> int:
    row = conn.execute(
        "SELECT MAX(run_id) FROM runs WHERE stage='correlation' AND interval=?", (interval,)
    ).fetchone()
    if not row or row[0] is None:
        raise SystemExit(f"No correlation run found for interval={interval!r}. Run --stage correlation first.")
    return row[0]


def best_oracle_sharpe(conn, assets_key: str, interval: str = INTERVAL,
                        backtest_period: str = BACKTEST_PERIOD):
    """MAX sharpe across strategies for this exact group's asset set, or None if oracle hasn't run it yet."""
    row = conn.execute(
        "SELECT MAX(sharpe) FROM oracle_results WHERE assets=? AND interval=? AND backtest_period=?",
        (assets_key, interval, backtest_period),
    ).fetchone()
    return row[0] if row else None


def select_prioritized_groups(conn, groups, stage, model_path, require_stage=None, require_since=None,
                               interval: str = INTERVAL, backtest_period: str = BACKTEST_PERIOD):
    """Filter `groups` (list of (group_id, assets) from kp.select_deduped_groups)
    to those not already done for (stage, model_path), optionally require an
    existing --require-stage row (joined against runs for --require-since
    freshness -- see run_base_priority.py's docstring on the 14-oracle-group
    hazard this guards against), then rank by best oracle Sharpe descending
    (None/unranked sorts last).

    Returns list of (group_id, assets, assets_key, sharpe).
    """
    prioritized = []
    for group_id, assets in groups:
        assets_key = ",".join(sorted(assets))
        already_done = conn.execute(
            "SELECT run_id FROM model_results WHERE assets=? AND interval=? "
            "AND backtest_period=? AND stage=? AND model_path IS ? LIMIT 1",
            (assets_key, interval, backtest_period, stage, model_path),
        ).fetchone()
        if already_done:
            continue
        if require_stage:
            sql = ("SELECT m.run_id FROM model_results m JOIN runs r ON r.run_id = m.run_id "
                   "WHERE m.assets=? AND m.interval=? AND m.backtest_period=? AND m.stage=?")
            params = [assets_key, interval, backtest_period, require_stage]
            if require_since:
                sql += " AND r.timestamp >= ?"
                params.append(require_since)
            if not conn.execute(sql + " LIMIT 1", params).fetchone():
                continue
        sharpe = best_oracle_sharpe(conn, assets_key, interval, backtest_period)
        prioritized.append((group_id, assets, assets_key, sharpe))

    prioritized.sort(key=lambda g: (g[3] is None, -(g[3] or 0.0)))
    return prioritized


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ── --gpu-workers auto: per-chunk sizing from free VRAM/RAM ─────────────────

def _ram_allows(avail_ram_mib, per_worker_rss_mib, cpu_workers_running):
    """How many per_worker_rss_mib-sized workers fit in available RAM after
    headroom and already-running CPU workers -- the RAM term shared by
    decide_gpu_workers() (fixed --gpu-workers <int> path) and
    decide_max_concurrent() (--gpu-workers auto's wave-size ceiling), so
    there is exactly one RAM-fits-N-workers formula in this file."""
    ram_for_workers = avail_ram_mib - RAM_HEADROOM_MIB - cpu_workers_running * per_worker_rss_mib
    return int(ram_for_workers // per_worker_rss_mib)


def decide_gpu_workers(free_vram_mib, avail_ram_mib, per_worker_vram_mib,
                        per_worker_rss_mib, max_workers, cpu_workers_running=0):
    """Pure sizing rule for the FIXED --gpu-workers <int> path. No I/O --
    every input is a plain number, so this is unit-testable without a GPU.
    cpu_workers_running is the number of --workers phase-2 processes already
    competing for RAM (nonzero only under --pipeline, where phase 1 and
    phase 2 run at once)."""
    vram_allows = int((free_vram_mib - VRAM_HEADROOM_MIB) // per_worker_vram_mib)
    ram_allows = _ram_allows(avail_ram_mib, per_worker_rss_mib, cpu_workers_running)
    return max(1, min(vram_allows, ram_allows, max_workers))


def decide_max_concurrent(avail_ram_mib, per_worker_rss_mib, physical_cores, cpu_workers_running=0):
    """Non-VRAM ceiling for pack_waves()'s wave size under --gpu-workers
    auto: min(physical cores, RAM-derived limit). VRAM itself is handled
    separately by pack_waves()'s own bin-packing against
    estimate_group_vram_mib() -- this only bounds concurrency the way
    physical CPU/RAM would, reusing decide_gpu_workers()'s RAM term
    (_ram_allows()) rather than a second RAM reader."""
    ram_allows = _ram_allows(avail_ram_mib, per_worker_rss_mib, cpu_workers_running)
    return max(1, min(ram_allows, physical_cores))


def _probe_free_vram_mib():
    """Free VRAM in MiB, via `nvidia-smi` ONLY. Returns None (not 0) when that
    fails -- no GPU on this box, CPU-only CI, etc -- so callers can tell
    "no GPU" apart from "GPU reports 0 MiB free".

    DO NOT probe with torch.cuda.mem_get_info() here, however tempting: it
    initialises a CUDA context *in this parent process*, and the phase-1
    prewarm pool is a fork-based ProcessPoolExecutor. A forked child of a
    CUDA-initialised parent dies with "Cannot re-initialize CUDA in forked
    subprocess", which is exactly what happened on 2026-08-31: every prewarm
    worker failed, the cache stayed empty, and phase 2 silently fell back to
    loading one model per worker until the GPU OOM'd (25 groups lost).
    Shelling out keeps this process CUDA-free so fork stays safe.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _probe_own_vram_mib(pids):
    """Sum of VRAM (MiB) actually used by OUR OWN subprocess pids, via
    `nvidia-smi --query-compute-apps`. This is the fix for the calibration
    bug in _phase1_with_vram_calibration(): the old approach sampled
    memory.free (whole-GPU) before/after and inferred usage from the delta,
    which is exactly what _probe_free_vram_mib()'s docstring warns not to
    trust for calibration -- it includes the display server AND any sibling
    session's own sweep. On this box that's not a hypothetical: multiple
    run_model_parallel.py sweeps run concurrently (see ps), and a 6GB card
    running 3 of them at once means the "before" and "min-during" whole-GPU
    readings are dominated by processes we don't control, so the delta
    computed from them was landing at <=0 on every single chunk -- the
    per_worker_vram_mib update was silently skipped every time, forever,
    not because of noise but because the two other sweeps kept the free
    number saturated regardless of what our own workers did.

    Querying --query-compute-apps instead gives a DIRECT, per-pid reading --
    no before/after delta needed at all, so it can't be swamped by anyone
    else's GPU usage. Returns 0.0 (not None) if nvidia-smi succeeds but none
    of `pids` show up yet (e.g. workers haven't loaded a model onto the GPU
    yet) -- that is a valid "no usage seen" reading, not a probe failure.
    Returns None only when nvidia-smi itself fails (no GPU, driver hiccup,
    etc), mirroring _probe_free_vram_mib()'s None-vs-0 contract.
    """
    if not pids:
        return 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except Exception:
        return None
    pid_set = set(pids)
    total = 0.0
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, used = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if pid in pid_set:
            total += used
    return total


def _resolve_gpu_workers_for_chunk(gpu_workers_arg, gpu_workers_max, per_worker_vram_mib,
                                    per_worker_rss_mib, cpu_workers_running=0,
                                    probe_vram_fn=_probe_free_vram_mib, probe_ram_mib_fn=None):
    """Returns (n, log_line). Called fresh immediately before each chunk's
    phase 1 -- free memory changes as other work starts/stops on this box
    (a sibling session can run its own GPU sweep concurrently). A fixed
    (non-"auto") gpu_workers_arg passes straight through unchanged.
    probe_vram_fn/probe_ram_mib_fn are injectable for tests."""
    if gpu_workers_arg != "auto":
        return gpu_workers_arg, f"gpu-workers={gpu_workers_arg} (fixed)"
    free_vram_mib = probe_vram_fn()
    if free_vram_mib is None:
        return 1, "gpu-workers=1 (auto: no GPU probe available -- torch+CUDA and nvidia-smi both failed)"
    ram_avail_mib = (probe_ram_mib_fn() if probe_ram_mib_fn is not None
                      else kp_predcache._read_mem_available_bytes() / (1024 ** 2))
    n = decide_gpu_workers(free_vram_mib, ram_avail_mib, per_worker_vram_mib, per_worker_rss_mib,
                            gpu_workers_max, cpu_workers_running)
    vram_allows = max(int((free_vram_mib - VRAM_HEADROOM_MIB) // per_worker_vram_mib), 0)
    ram_allows = max(int((ram_avail_mib - RAM_HEADROOM_MIB - cpu_workers_running * per_worker_rss_mib)
                          // per_worker_rss_mib), 0)
    return n, (f"gpu-workers={n} (auto: vram_free={free_vram_mib:.0f}MiB -> {vram_allows}, "
               f"ram_avail={ram_avail_mib:.0f}MiB -> {ram_allows}, max={gpu_workers_max})")


class _MinSampler:
    """ponytail: coarse ~1s-poll extreme-value tracker used to self-calibrate
    the per-worker VRAM/RAM estimates chunk-to-chunk. Not an NVML event hook
    -- good enough to nudge the next chunk's --gpu-workers auto decision, not
    a precise profiler. Upgrade path: NVML polling thread if this ever
    proves too noisy in practice.

    Tracks the running MINIMUM by default (used for RAM: "available" is a
    free-quantity, so the low point is the peak usage). Pass keep_max=True
    to track the running MAXIMUM instead (used for VRAM since 2026-08-31:
    _probe_own_vram_mib() reports a used-quantity directly, so the high
    point -- not a before/after delta -- is the peak usage; see that
    function's docstring for why the delta approach this replaced was
    unreliable with concurrent GPU users on the box)."""

    def __init__(self, probe_fn, interval=1.0, keep_max=False):
        self._probe_fn = probe_fn
        self._interval = interval
        self._keep_max = keep_max
        self.extreme_value = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            v = self._probe_fn()
            if v is not None:
                if self.extreme_value is None:
                    self.extreme_value = v
                elif self._keep_max and v > self.extreme_value:
                    self.extreme_value = v
                elif not self._keep_max and v < self.extreme_value:
                    self.extreme_value = v
            self._stop.wait(self._interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def _phase1_with_vram_calibration(gpu_workers, per_worker_vram_mib, per_worker_rss_mib,
                                   auto, run_phase1_fn, probe_own_vram_fn=_probe_own_vram_mib,
                                   sample_interval=1.0):
    """Runs run_phase1_fn(pool_pids) -- a one-arg closure around run_phase1()
    that must extend the given list with the pids of whatever GPU worker
    process(es) it spawns (run_phase1's own pids_out= param does this) -- and,
    only when auto, samples per-process VRAM usage + free RAM around the call
    to refine per_worker_vram_mib/per_worker_rss_mib from what THIS phase
    actually used, so the NEXT chunk's decide_gpu_workers() call sizes better
    than the fixed seed. No sampling overhead when auto is off.

    VRAM is measured directly via probe_own_vram_fn(pids) (see
    _probe_own_vram_mib's docstring for why this replaced a memory.free
    before/after delta). RAM stays on the free-quantity delta pattern since
    kp_predcache._read_mem_available_bytes() is a whole-box reading with no
    per-pid equivalent available.

    sample_interval is the _MinSampler poll period (default 1.0s in
    production); tests pass a much smaller value so a fast fake
    run_phase1_fn still gets sampled at least once.

    Returns (run_phase1 result, new_per_worker_vram_mib, new_per_worker_rss_mib).
    """
    pool_pids = []
    if not auto:
        return run_phase1_fn(pool_pids), per_worker_vram_mib, per_worker_rss_mib

    ram_before_mib = kp_predcache._read_mem_available_bytes() / (1024 ** 2)
    vram_sampler = _MinSampler(lambda: probe_own_vram_fn(pool_pids), interval=sample_interval, keep_max=True)
    ram_sampler = _MinSampler(lambda: kp_predcache._read_mem_available_bytes() / (1024 ** 2), interval=sample_interval)
    with vram_sampler, ram_sampler:
        result = run_phase1_fn(pool_pids)

    if vram_sampler.extreme_value and gpu_workers > 0:  # 0.0 (no usage seen) is falsy -- correctly skipped
        per_worker_vram_mib = max(vram_sampler.extreme_value / gpu_workers, 256.0)
    if ram_sampler.extreme_value is not None and gpu_workers > 0:
        used = ram_before_mib - ram_sampler.extreme_value
        if used > 0:
            per_worker_rss_mib = max(used / gpu_workers, 256.0)
    return result, per_worker_vram_mib, per_worker_rss_mib


# ── Phase 1: GPU prediction-cache prewarm ───────────────────────────────────

def enumerate_backtest_dates(data_dict, lookback):
    """Reimplements KairosOrchestrator.run_backtest's date/history
    enumeration (strategy/kairos_orchestrator.py:925-961) verbatim, without
    touching the GPU -- this is what the backtest subprocess will iterate
    over, so phase 1 must visit the exact same (date, histories) sequence
    for its prediction-cache keys to land the hits phase 2 depends on.

    Yields (date, histories) where histories is {symbol: df sliced to <=date}.
    """
    all_dates = set()
    for df in data_dict.values():
        all_dates.update(df.index[lookback:])
    for date in sorted(all_dates):
        histories = {}
        for symbol, df in data_dict.items():
            mask = df.index <= date
            if mask.sum() < lookback:
                continue
            histories[symbol] = df[mask]
        if not histories:
            continue
        yield date, histories


def _prewarm_group(assets, model_path, interval, backtest_period, pred_samples):
    """Populate the shared kairos_predcache for one group across its full
    backtest date range. Mirrors strategy/kairos_strategies.py's __main__
    block (data_dict construction, ~line 1158) and
    kairos_papertrade.prewarm_prediction_cache()'s _sweep_unit() (single
    fetch+check+predict pass, periodic gc.collect()) -- KairosSettings must
    be configured identically to the backtest subprocess before calling
    is_batch_cached/predict_all_batch, since _shared_cache_key reads
    KairosSettings.lookback/.interval/.pred_samples.

    Requires KAIROS_PRED_CACHE_DIR already set in the environment (the
    caller's job, so it's set once per process/worker, not per group).

    Returns (hits, misses).
    """
    import kairos_strategies as ks

    ks.KairosSettings.interval = interval
    ks.KairosSettings.backtest_period = backtest_period
    ks.KairosSettings.pred_samples = pred_samples
    lookback = ks.KairosSettings.lookback

    n_bars = ks._period_to_bars(backtest_period, interval)
    data_dict = {
        sym: ks.fetch_data_raw(sym, lookback, min_bars=lookback + n_bars).tail(lookback + n_bars)
        for sym in assets
    }

    hits = misses = 0
    for i, (_date, histories) in enumerate(enumerate_backtest_dates(data_dict, lookback), start=1):
        if ks.is_batch_cached(histories, model_path=model_path):
            hits += 1
        else:
            ks.predict_all_batch(histories, model_path=model_path, build_distributions=False)
            misses += 1
        if i % _PREWARM_GC_INTERVAL == 0:
            gc.collect()
    return hits, misses


def _prewarm_worker(group_id, assets_key, assets, model_path, interval, backtest_period, pred_samples,
                     cache_dir, cache_max_bytes):
    """Runs in a forked --gpu-workers>1 process. Must stay at module level (pickling)."""
    os.environ.update(_THREAD_ENV)
    os.environ["KAIROS_PRED_CACHE_DIR"] = cache_dir
    if cache_max_bytes:
        os.environ["KAIROS_PRED_CACHE_MAX_BYTES"] = str(cache_max_bytes)
    try:
        hits, misses = _prewarm_group(assets, model_path, interval, backtest_period, pred_samples)
        return (group_id, assets_key, "done", hits, misses, None)
    except Exception as exc:
        return (group_id, assets_key, "fail", 0, 0, str(exc))


class PrewarmWhollyFailed(RuntimeError):
    """Every group in a chunk failed to prewarm -- the split is not working."""


def _abort_if_prewarm_wholly_failed(chunk, hits, misses, failures):
    """Stop the run when a chunk prewarmed NOTHING.

    Phase 2 is designed to survive a per-group prewarm miss by loading the
    model itself, which is the right call for one bad group. But when the
    prewarm fails for the WHOLE chunk, that fallback stops being a safety net
    and becomes a hazard: every phase-2 worker loads its own model
    simultaneously, and N concurrent models OOM the GPU. That is not
    hypothetical -- on 2026-08-31 a CUDA-in-forked-parent bug failed 100% of
    prewarms, and the run ground on for three chunks losing 25 groups to
    CUDA OOM instead of stopping at the first sign.

    Failing loudly here costs one chunk. Absorbing it cost 25 groups and an
    hour of GPU time.
    """
    if failures and hits == 0 and misses == 0 and len(failures) >= len(chunk):
        raise PrewarmWhollyFailed(
            f"prewarm failed for all {len(chunk)} groups in this chunk and cached "
            f"nothing (hits=0, misses=0). Phase 2 would fall back to one model "
            f"load per worker and OOM the GPU, so stopping instead.\n"
            f"First error: {failures[0][1]}\n"
            f"Groups already completed are committed; re-run to resume."
        )


def run_phase1(chunk, model_path, interval, backtest_period, pred_samples,
               cache_dir, cache_max_bytes, gpu_workers, force_pool=False, pids_out=None):
    """chunk: list of (group_id, assets, assets_key, sharpe). Returns (hits, misses, failures).

    force_pool=True always dispatches through ProcessPoolExecutor, even for
    gpu_workers<=1 -- required under --pipeline so phase 1's Python work runs
    in a worker process, not inline on the thread that's supposed to let
    phase 2 run concurrently (an inline call would hold the GIL on the main
    thread's ThreadPoolExecutor worker and block phase 2's own submits).

    pids_out, if given, is extended with the pid(s) actually doing the GPU
    work -- the pool's forked child pids, or (serial path) this process's own
    pid. _phase1_with_vram_calibration() uses this to filter its VRAM
    sampling to processes we actually spawned, see _probe_own_vram_mib()."""
    total_hits = total_misses = 0
    failures = []
    if gpu_workers <= 1 and not force_pool:
        # Fully serial (default): no subprocess/pickling overhead, one GPU
        # user at a time -- the safe default for a single GPU box.
        if pids_out is not None:
            pids_out.append(os.getpid())
        for group_id, assets, assets_key, _sharpe in chunk:
            try:
                hits, misses = _prewarm_group(assets, model_path, interval, backtest_period, pred_samples)
                total_hits += hits
                total_misses += misses
            except Exception as exc:
                failures.append((assets_key, str(exc)))
    else:
        with ProcessPoolExecutor(max_workers=max(gpu_workers, 1)) as pool:
            futures = {
                pool.submit(_prewarm_worker, gid, ak, assets, model_path, interval, backtest_period,
                            pred_samples, cache_dir, cache_max_bytes): ak
                for gid, assets, ak, _sharpe in chunk
            }
            if pids_out is not None:
                pids_out.extend(getattr(pool, "_processes", {}).keys())
            for fut in as_completed(futures):
                _gid, ak, status, hits, misses, err = fut.result()
                if status == "fail":
                    failures.append((ak, err))
                else:
                    total_hits += hits
                    total_misses += misses
    return total_hits, total_misses, failures


def _calibrate_wave(wave, model_id, observed_total_peak_mib):
    """Feeds one wave's observed total per-pid VRAM peak back into
    calibrate_vram_table(), attributing it to the wave's group size -- only
    when every group in the wave shares the same n_assets (an uneven wave's
    average total/len doesn't attribute cleanly to either size, so it's
    skipped rather than guessed)."""
    if not wave or not observed_total_peak_mib:
        return
    sizes = {len(g[1]) for g in wave}
    if len(sizes) != 1:
        return
    n = next(iter(sizes))
    calibrate_vram_table(model_id, n, observed_total_peak_mib / len(wave))


def run_phase1_packed(chunk, model_path, interval, backtest_period, pred_samples,
                       cache_dir, cache_max_bytes, budget_mib, max_concurrent,
                       chunk_label="?", n_chunks="?",
                       probe_own_vram_fn=_probe_own_vram_mib, sample_interval=1.0,
                       calibrate=True, pids_out=None):
    """--gpu-workers auto phase-1 driver: bin-pack `chunk` into VRAM-fitting
    waves (pack_waves()) and run each wave through run_phase1(), one
    ProcessPoolExecutor(max_workers=len(wave)) per wave (force_pool=True
    unconditionally -- packed waves always dispatch through the pool, even a
    size-1 wave, so this stays safe to call from inside --pipeline's
    background-thread prefetch same as the fixed-int path's force_pool
    requirement). Waves run one after another, never overlapping each other.
    Logs each wave so the packing decision is auditable:
      [chunk 3/10] wave 1/3: 3 groups, est 4044/5296 MiB (sizes 1,1,1)

    When calibrate, samples each wave's own peak per-pid VRAM (same
    _MinSampler/_probe_own_vram_mib mechanism as
    _phase1_with_vram_calibration()) and feeds it into calibrate_vram_table()
    for waves where every group is the same size.

    Returns (hits, misses, failures) -- same shape as run_phase1()."""
    waves = pack_waves(chunk, model_path, budget_mib, max_concurrent)
    total_hits = total_misses = 0
    failures = []
    for wi, wave in enumerate(waves, start=1):
        sizes = ",".join(str(len(g[1])) for g in wave)
        est = sum(estimate_group_vram_mib(model_path, len(g[1])) for g in wave)
        print(f"[chunk {chunk_label}/{n_chunks}] wave {wi}/{len(waves)}: {len(wave)} groups, "
              f"est {est:.0f}/{budget_mib:.0f} MiB (sizes {sizes})")

        wave_pids = []
        if calibrate:
            sampler = _MinSampler(lambda: probe_own_vram_fn(wave_pids), interval=sample_interval, keep_max=True)
            with sampler:
                hits, misses, wave_failures = run_phase1(
                    wave, model_path, interval, backtest_period, pred_samples, cache_dir, cache_max_bytes,
                    gpu_workers=len(wave), force_pool=True, pids_out=wave_pids,
                )
            _calibrate_wave(wave, model_path, sampler.extreme_value)
        else:
            hits, misses, wave_failures = run_phase1(
                wave, model_path, interval, backtest_period, pred_samples, cache_dir, cache_max_bytes,
                gpu_workers=len(wave), force_pool=True, pids_out=wave_pids,
            )

        if pids_out is not None:
            pids_out.extend(wave_pids)
        total_hits += hits
        total_misses += misses
        failures.extend(wave_failures)
    return total_hits, total_misses, failures


def _phase1_auto_for_chunk(chunk, model_path, interval, backtest_period, pred_samples, cache_dir,
                            cache_max_bytes, gpu_workers_max, per_worker_rss_mib, physical_cores,
                            cpu_workers_running, chunk_label, n_chunks,
                            probe_vram_fn=_probe_free_vram_mib, probe_ram_mib_fn=None,
                            sample_interval=1.0, pids_out=None):
    """--gpu-workers auto phase 1 for one chunk: probe free VRAM/RAM the
    existing way (nvidia-smi only -- see _probe_free_vram_mib()'s docstring
    for why torch must never do this probe), derive a wave-packing budget
    (free VRAM - VRAM_HEADROOM_MIB) and max_concurrent (RAM/physical-cores
    ceiling via decide_max_concurrent()), then run_phase1_packed().

    Also refines per_worker_rss_mib from this chunk's own RAM delta for the
    next chunk's decide_max_concurrent() call -- the same RAM-delta pattern
    _phase1_with_vram_calibration() uses, kept separate here because that
    helper's single-pool-size VRAM math doesn't apply once one chunk can run
    through several differently-sized wave pools; VRAM calibration instead
    happens per-wave inside run_phase1_packed() via calibrate_vram_table().

    Returns (hits, misses, failures, new_per_worker_rss_mib)."""
    read_ram_mib = probe_ram_mib_fn or (lambda: kp_predcache._read_mem_available_bytes() / (1024 ** 2))
    free_vram_mib = probe_vram_fn()
    if free_vram_mib is None:
        print(f"[chunk {chunk_label}/{n_chunks}] gpu-workers=1 (auto: no GPU probe available -- nvidia-smi failed)")
        hits, misses, failures = run_phase1(
            chunk, model_path, interval, backtest_period, pred_samples, cache_dir, cache_max_bytes,
            gpu_workers=1, force_pool=True, pids_out=pids_out,
        )
        return hits, misses, failures, per_worker_rss_mib

    ram_avail_mib = read_ram_mib()
    budget_mib = max(free_vram_mib - VRAM_HEADROOM_MIB, 0.0)
    max_concurrent = min(
        decide_max_concurrent(ram_avail_mib, per_worker_rss_mib, physical_cores, cpu_workers_running),
        gpu_workers_max,
    )

    ram_before_mib = ram_avail_mib
    ram_sampler = _MinSampler(read_ram_mib, interval=sample_interval)
    with ram_sampler:
        hits, misses, failures = run_phase1_packed(
            chunk, model_path, interval, backtest_period, pred_samples, cache_dir, cache_max_bytes,
            budget_mib, max_concurrent, chunk_label=chunk_label, n_chunks=n_chunks,
            sample_interval=sample_interval, pids_out=pids_out,
        )

    new_rss = per_worker_rss_mib
    if ram_sampler.extreme_value is not None and max_concurrent > 0:
        used = ram_before_mib - ram_sampler.extreme_value
        if used > 0:
            new_rss = max(used / max_concurrent, 256.0)
    return hits, misses, failures, new_rss


# ── Phase 2: CPU-parallel model-stage run ───────────────────────────────────

def _run_stage_worker(group_id, assets, assets_key, stage, model_path, interval, backtest_period,
                       pred_samples, extra_env):
    """Runs in a forked worker process. Must stay at module level (pickling).
    Each worker opens its own sqlite connection -- never share one across
    processes (see scripts/run_oracle_dedup.py's _connect())."""
    os.environ.update(_THREAD_ENV)
    import kairos_pipeline as _kp
    conn = _kp.get_connection(_kp.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        run_id = _kp.run_stage_model(
            conn, stage, assets, interval=interval, backtest_period=backtest_period,
            pred_samples=pred_samples, model_path=model_path, extra_env=extra_env,
        )
        return (group_id, assets_key, "done", run_id, None)
    except Exception as exc:
        return (group_id, assets_key, "fail", None, str(exc))
    finally:
        conn.close()


def run_phase2(chunk, stage, model_path, interval, backtest_period, pred_samples, extra_env, workers):
    """chunk: list of (group_id, assets, assets_key, sharpe). Returns (done, failed, failures)."""
    done = failed = 0
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_stage_worker, gid, assets, ak, stage, model_path, interval, backtest_period,
                        pred_samples, extra_env): ak
            for gid, assets, ak, _sharpe in chunk
        }
        for fut in as_completed(futures):
            _gid, ak, status, _run_id, err = fut.result()
            if status == "done":
                done += 1
            else:
                failed += 1
                failures.append((ak, err))
    return done, failed, failures


def run_pipelined(chunks, phase1_fn, phase2_fn, on_prefetch_fail=None):
    """Generic chunk-pipelining scheduler behind --pipeline (see module
    docstring). Owns ONLY the overlap scheduling + prefetch-failure
    fallback -- phase1_fn(chunk)/phase2_fn(chunk) are opaque callables, so
    this is unit-testable with plain stub functions and no GPU/DB/model_path
    involved (the real phase1_fn/phase2_fn main() wires in close over
    run_phase1/run_phase2 and the CLI's state).

    on_prefetch_fail(chunk, exc), if given, runs synchronously right before
    the inline fallback prewarm for that chunk -- callers use it for logging.

    Yields, per chunk in order: (chunk, phase1_result, phase2_result,
    phase1_wait_s, phase2_s, wall_s, fell_back).
    """
    with ThreadPoolExecutor(max_workers=1) as pre:
        prefetch = None
        for i, chunk in enumerate(chunks):
            wall_t0 = time.time()
            fell_back = False
            if prefetch is None:
                t0 = time.time()
                phase1_result = phase1_fn(chunk)
                phase1_wait_s = time.time() - t0
            else:
                t0 = time.time()
                try:
                    phase1_result = prefetch.result()
                except Exception as exc:
                    fell_back = True
                    if on_prefetch_fail is not None:
                        on_prefetch_fail(chunk, exc)
                    phase1_result = phase1_fn(chunk)
                phase1_wait_s = time.time() - t0

            nxt = chunks[i + 1] if i + 1 < len(chunks) else None
            prefetch = pre.submit(phase1_fn, nxt) if nxt is not None else None

            t2_0 = time.time()
            phase2_result = phase2_fn(chunk)
            phase2_s = time.time() - t2_0
            wall_s = time.time() - wall_t0

            yield chunk, phase1_result, phase2_result, phase1_wait_s, phase2_s, wall_s, fell_back


# ── --control-file: live workers/gpu_workers/chunk_size rebalancing ────────

def _load_control_overrides(path, workers, gpu_workers, chunk_size, cpu_count):
    """Re-read before each chunk. Returns (workers, gpu_workers, chunk_size,
    changes) where changes is a list of "key old -> new" strings for values
    that actually changed -- empty when nothing changed, INCLUDING a missing
    file, so callers can stay silent in the normal (no control file) case.

    Never raises. A missing file is normal and silent. Malformed JSON, a
    non-object top level, wrong types, or an out-of-range count are each a
    single WARNING line naming the problem, keeping the PRIOR value for that
    key -- a typo'd control file must not kill an hours-long sweep."""
    if not os.path.isfile(path):
        return workers, gpu_workers, chunk_size, []
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[control-file] WARNING: could not parse {path}: {exc} -- keeping current values")
        return workers, gpu_workers, chunk_size, []
    if not isinstance(data, dict):
        print(f"[control-file] WARNING: {path} must contain a JSON object, got "
              f"{type(data).__name__} -- keeping current values")
        return workers, gpu_workers, chunk_size, []

    def _valid_count(v):
        return isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= cpu_count

    changes = []

    if "workers" in data:
        v = data["workers"]
        if not _valid_count(v):
            print(f"[control-file] WARNING: 'workers'={v!r} invalid (must be an integer "
                  f"1..{cpu_count}) -- keeping workers={workers}")
        elif v != workers:
            changes.append(f"workers {workers} -> {v}")
            workers = v

    if "gpu_workers" in data:
        v = data["gpu_workers"]
        if v == "auto" or _valid_count(v):
            if v != gpu_workers:
                changes.append(f"gpu_workers {gpu_workers} -> {v}")
                gpu_workers = v
        else:
            print(f"[control-file] WARNING: 'gpu_workers'={v!r} invalid (must be an integer "
                  f"1..{cpu_count} or 'auto') -- keeping gpu_workers={gpu_workers}")

    if "chunk_size" in data:
        v = data["chunk_size"]
        if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
            print(f"[control-file] WARNING: 'chunk_size'={v!r} invalid (must be a positive "
                  f"integer) -- keeping chunk_size={chunk_size}")
        elif v != chunk_size:
            changes.append(f"chunk_size {chunk_size} -> {v}")
            chunk_size = v

    return workers, gpu_workers, chunk_size, changes


def _warn_if_over_physical(workers, gpu_workers, physical, context):
    """Advisory only -- mirrors the pre-existing --pipeline startup check.
    Skipped when gpu_workers is "auto": the real per-chunk count varies and
    is whatever decide_gpu_workers() just sized it to, not this value."""
    if gpu_workers == "auto":
        return
    total = workers + gpu_workers
    if total > physical:
        print(f"WARNING: {context} workers ({workers}) + gpu_workers ({gpu_workers}) = {total} "
              f"processes > {physical} physical cores. Peak RAM may be tight -- consider lower values.")


# ── CLI ──────────────────────────────────────────────────────────────────

def _gpu_workers_arg(v):
    if v == "auto":
        return "auto"
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--gpu-workers must be an integer or 'auto', got {v!r}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("max_hours", nargs="?", type=float, default=8.0,
                    help="Time budget in hours (default: 8.0). Only gates STARTING a new chunk -- "
                         "an in-progress chunk is never killed mid-run.")
    p.add_argument("--model", default=None,
                    help="Registry short name (base|small|mini), HF repo id, or local checkpoint "
                         "path. Default: base's model_path (None).")
    p.add_argument("--stage", default="base", help="model_results stage to write (default: base).")
    p.add_argument("--limit", type=int, default=None, help="Cap the number of groups run this invocation.")
    p.add_argument("--require-stage", default=None,
                    help="Only run groups that already have a model_results row for this stage.")
    p.add_argument("--require-since", default=None, metavar="YYYY-MM-DD",
                    help="With --require-stage, require that row written on/after this runs.timestamp date.")
    p.add_argument("--assets-file", default=None, metavar="FILE",
                    help="Explicit group list (one comma-joined asset set per line, '#' comments "
                         "allowed), bypassing select_deduped_groups' priority ordering. Use it to "
                         "pin an identical group set across stages -- the eligible pool differs per "
                         "stage, so priority selection would give each model different groups.")
    p.add_argument("--workers", type=int, default=8, help="Phase-2 CPU worker processes (default: 8).")
    p.add_argument("--gpu-workers", type=_gpu_workers_arg, default=1,
                    help="Phase-1 GPU worker processes (default: 1 -- fully serial). Pass 'auto' to "
                         "size per chunk from free VRAM/RAM (see --gpu-workers-max, decide_gpu_workers()).")
    p.add_argument("--gpu-workers-max", type=int, default=4,
                    help="Ceiling for --gpu-workers auto (default: 4).")
    p.add_argument("--chunk-size", type=int, default=16, help="Groups per prewarm/run chunk (default: 16).")
    p.add_argument("--cache-max-gb", type=float, default=8.0,
                    help="Shared prediction-cache disk budget in GiB (default: 8).")
    p.add_argument("--cache-dir", default=None,
                    help="Shared prediction-cache directory (default: data/predcache_sweep/).")
    p.add_argument("--pipeline", action="store_true",
                    help="Overlap chunk N+1's phase 1 (GPU prewarm) with chunk N's phase 2 (CPU "
                         "replay). Default off (current sequential behaviour). See module docstring.")
    p.add_argument("--control-file", default=DEFAULT_CONTROL_FILE,
                    help="JSON file re-read before each chunk to rebalance --workers/--gpu-workers/"
                         "--chunk-size live, without restarting the sweep (default: "
                         "data/sweep_control.json). Missing file is normal/silent; a malformed one "
                         "warns and keeps the current values. See module docstring for the format.")
    return p.parse_args()


def main():
    args = _parse_args()
    model_path = resolve_model(args.model)["model_id"] if args.model else None
    deadline = time.time() + args.max_hours * 3600

    # Respect an already-set KAIROS_PRED_CACHE_DIR (external/multi-session
    # choice) ahead of --cache-dir/default, mirroring
    # kairos_papertrade._ensure_pred_cache_dir_env()'s precedent.
    cache_dir = os.environ.get("KAIROS_PRED_CACHE_DIR") or args.cache_dir or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["KAIROS_PRED_CACHE_DIR"] = cache_dir
    cache_max_bytes = int(args.cache_max_gb * (1024 ** 3))
    os.environ.setdefault("KAIROS_PRED_CACHE_MAX_BYTES", str(cache_max_bytes))
    extra_env = dict(_THREAD_ENV, KAIROS_PRED_CACHE_DIR=cache_dir,
                      KAIROS_PRED_CACHE_MAX_BYTES=os.environ["KAIROS_PRED_CACHE_MAX_BYTES"])

    conn = kp.get_connection(kp.DB_PATH)
    if args.assets_file:
        # Explicit group list, one comma-joined asset set per line, bypassing
        # select_deduped_groups' priority ordering. Mirrors the same option on
        # scripts/run_oracle_dedup.py. Needed to pin an IDENTICAL group set
        # across several stages: the eligible pool differs per stage (a group
        # is eligible if it lacks a row for THAT stage), so priority selection
        # would hand each model a different 20 and quietly ruin a comparison.
        with open(args.assets_file) as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        groups = [(-1, ln.split(",")) for ln in lines]
        print(f"Group list: {len(groups)} from {args.assets_file} (priority ordering bypassed)")
    else:
        correlation_run_id = latest_correlation_run_id(conn, INTERVAL)
        groups = kp.select_deduped_groups(conn, correlation_run_id)
    prioritized = select_prioritized_groups(
        conn, groups, args.stage, model_path, args.require_stage, args.require_since,
    )
    conn.close()
    if args.limit is not None:
        prioritized = prioritized[: args.limit]

    _src = (f"from {args.assets_file}" if args.assets_file
            else f"correlation run_id={correlation_run_id}")
    print(f"Unprocessed groups: {len(prioritized)} (of {len(groups)} total, "
          f"{_src}). Stage={args.stage!r} model_path={model_path!r}. "
          f"Budget: {args.max_hours}h. Chunk size: {args.chunk_size}. Cache dir: {cache_dir}. "
          f"Control file: {args.control_file}")

    done_total = stage_failed_total = prewarm_failed_total = 0
    per_worker_vram_mib = DEFAULT_PER_WORKER_VRAM_MIB
    per_worker_rss_mib = DEFAULT_PER_WORKER_RSS_MIB
    cpu_count = os.cpu_count() or 1
    physical = cpu_count // 2

    # Mutable, control-file-adjustable balance. Read via these, not args.*,
    # everywhere below -- args.* stays the launch-time value only.
    current_workers = args.workers
    current_gpu_workers = args.gpu_workers
    current_chunk_size = args.chunk_size

    def _apply_control(chunk_label):
        nonlocal current_workers, current_gpu_workers, current_chunk_size
        current_workers, current_gpu_workers, current_chunk_size, changes = _load_control_overrides(
            args.control_file, current_workers, current_gpu_workers, current_chunk_size, cpu_count,
        )
        if changes:
            print(f"[chunk {chunk_label}] control-file: " + ", ".join(changes))
            _warn_if_over_physical(current_workers, current_gpu_workers, physical, "control-file:")

    def _resolve_gw():
        # Under --pipeline, phase 1's prewarm runs concurrently with phase 2's
        # --workers pool, so the RAM term of --gpu-workers auto's sizing must
        # treat those workers as already-committed RAM, not free RAM.
        cpu_workers_running = current_workers if args.pipeline else 0
        return _resolve_gpu_workers_for_chunk(
            current_gpu_workers, args.gpu_workers_max, per_worker_vram_mib, per_worker_rss_mib,
            cpu_workers_running,
        )

    if args.pipeline:
        # auto mode's actual gpu-workers varies per chunk; use the ceiling for this startup check.
        worst_case_gw = args.gpu_workers_max if args.gpu_workers == "auto" else args.gpu_workers
        _warn_if_over_physical(args.workers, worst_case_gw, physical, "--pipeline startup:")

        # ponytail: chunk_size is NOT live-adjustable under --pipeline -- the chunk
        # list is built once, up front, because run_pipelined()'s lookahead prefetch
        # needs random access to "chunk i+1" one step before chunk i finishes. A
        # control-file chunk_size edit is still parsed/validated (so a typo still
        # just warns, never crashes) but only takes effect for workers/gpu_workers;
        # rebuilding the chunk list live would need run_pipelined() to pull chunks
        # from a queue instead of indexing a fixed list -- add that if chunk_size
        # needs to move mid-pipelined-run.
        chunks = list(_chunked(prioritized, current_chunk_size))
        phase1_calls = 0

        def _phase1_call(chunk_):
            nonlocal per_worker_vram_mib, per_worker_rss_mib, phase1_calls
            phase1_calls += 1
            _apply_control(f"~{phase1_calls}")
            if current_gpu_workers == "auto":
                # Under --pipeline, phase 1 runs concurrently with phase 2's
                # --workers pool, so max_concurrent's RAM term must treat
                # those workers as already-committed RAM (mirrors the old
                # _resolve_gw()'s cpu_workers_running handling).
                hits, misses, p1_failures, per_worker_rss_mib = _phase1_auto_for_chunk(
                    chunk_, model_path, INTERVAL, BACKTEST_PERIOD, PRED_SAMPLES, cache_dir, cache_max_bytes,
                    args.gpu_workers_max, per_worker_rss_mib, physical, cpu_workers_running=current_workers,
                    chunk_label=f"~{phase1_calls}", n_chunks=len(chunks),
                )
                return hits, misses, p1_failures
            gw, gw_log = _resolve_gw()
            print(f"  [prewarm] groups={len(chunk_)} {gw_log}")
            result, per_worker_vram_mib, per_worker_rss_mib = _phase1_with_vram_calibration(
                gw, per_worker_vram_mib, per_worker_rss_mib, False,
                lambda pids: run_phase1(chunk_, model_path, INTERVAL, BACKTEST_PERIOD, PRED_SAMPLES,
                                         cache_dir, cache_max_bytes, gw, force_pool=True, pids_out=pids),
            )
            return result

        def _phase2_call(chunk_):
            return run_phase2(chunk_, args.stage, model_path, INTERVAL, BACKTEST_PERIOD, PRED_SAMPLES,
                               extra_env, current_workers)

        def _on_prefetch_fail(chunk_, exc):
            print(f"  [prefetch FAIL] {exc} -- falling back to inline prewarm for this chunk")

        ci = 0
        gen = run_pipelined(chunks, _phase1_call, _phase2_call, on_prefetch_fail=_on_prefetch_fail)
        try:
            for chunk, phase1_result, phase2_result, phase1_wait_s, phase2_s, wall_s, fell_back in gen:
                ci += 1
                hits, misses, p1_failures = phase1_result
                done, stage_failed, p2_failures = phase2_result

                done_total += done
                stage_failed_total += stage_failed
                prewarm_failed_total += len(p1_failures)

                for ak, err in p1_failures:
                    print(f"  [prewarm FAIL] {ak}: {err}")
                for ak, err in p2_failures:
                    print(f"  [stage FAIL] {ak}: {err}")

                _abort_if_prewarm_wholly_failed(chunk, hits, misses, p1_failures)

                fb_note = " prefetch_fallback=1" if fell_back else ""
                print(f"[chunk {ci}/{len(chunks)}] groups={len(chunk)} prewarm_hits={hits} "
                      f"prewarm_misses={misses} phase1_wait_s={phase1_wait_s:.1f} "
                      f"phase2_s={phase2_s:.1f} wall_s={wall_s:.1f}{fb_note} "
                      f"done_total={done_total} stage_failed_total={stage_failed_total} "
                      f"prewarm_failed_total={prewarm_failed_total}")

                # Checked AFTER each chunk (not before, as the sequential path does) because
                # the next chunk's prewarm is already submitted in the background by the time
                # this chunk's results are in hand -- there's no cheap earlier point to gate on.
                if time.time() >= deadline:
                    n_remaining = sum(len(c) for c in chunks[ci:])
                    print(f"\n[budget] {args.max_hours}h elapsed after chunk {ci}/{len(chunks)}. "
                          f"{n_remaining} groups remain for next invocation.")
                    break
        finally:
            gen.close()  # waits out any in-flight prefetch before returning
    else:
        remaining = list(prioritized)
        ci = 0
        while remaining:
            if time.time() >= deadline:
                print(f"\n[budget] {args.max_hours}h elapsed, stopping before chunk {ci + 1}. "
                      f"{len(remaining)} groups remain for next invocation.")
                break
            ci += 1
            _apply_control(ci)
            chunk, remaining = remaining[:current_chunk_size], remaining[current_chunk_size:]
            est_total = ci + math.ceil(len(remaining) / current_chunk_size) if remaining else ci

            gpu_auto = current_gpu_workers == "auto"
            t0 = time.time()
            if gpu_auto:
                hits, misses, p1_failures, per_worker_rss_mib = _phase1_auto_for_chunk(
                    chunk, model_path, INTERVAL, BACKTEST_PERIOD, PRED_SAMPLES, cache_dir, cache_max_bytes,
                    args.gpu_workers_max, per_worker_rss_mib, physical, cpu_workers_running=0,
                    chunk_label=ci, n_chunks=est_total,
                )
            else:
                gw, gw_log = _resolve_gw()
                (hits, misses, p1_failures), per_worker_vram_mib, per_worker_rss_mib = _phase1_with_vram_calibration(
                    gw, per_worker_vram_mib, per_worker_rss_mib, False,
                    lambda pids: run_phase1(chunk, model_path, INTERVAL, BACKTEST_PERIOD, PRED_SAMPLES,
                                             cache_dir, cache_max_bytes, gw, pids_out=pids),
                )
            t1 = time.time()
            # A prewarm failure for a SOME groups doesn't block phase 2 -- those
            # groups' stage subprocesses just fall back to loading the model
            # themselves. A prewarm failure for ALL of them is a different animal
            # and must not be absorbed the same way: see the guard's docstring.
            for ak, err in p1_failures:
                print(f"  [prewarm FAIL] {ak}: {err}")
            _abort_if_prewarm_wholly_failed(chunk, hits, misses, p1_failures)
            done, stage_failed, p2_failures = run_phase2(
                chunk, args.stage, model_path, INTERVAL, BACKTEST_PERIOD, PRED_SAMPLES,
                extra_env, current_workers,
            )
            t2 = time.time()

            done_total += done
            stage_failed_total += stage_failed
            prewarm_failed_total += len(p1_failures)

            for ak, err in p2_failures:
                print(f"  [stage FAIL] {ak}: {err}")

            print(f"[chunk {ci}/{est_total}] groups={len(chunk)} prewarm_hits={hits} "
                  f"prewarm_misses={misses} phase1_s={t1 - t0:.1f} phase2_s={t2 - t1:.1f} "
                  f"done_total={done_total} stage_failed_total={stage_failed_total} "
                  f"prewarm_failed_total={prewarm_failed_total}")

    print(f"\nParallel {args.stage} sweep done: {done_total} run, {stage_failed_total} stage failures, "
          f"{prewarm_failed_total} prewarm failures, total={len(prioritized)}.")


if __name__ == "__main__":
    main()
