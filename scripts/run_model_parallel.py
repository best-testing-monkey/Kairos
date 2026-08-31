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
"""
from __future__ import annotations

import argparse
import gc
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
_PREWARM_GC_INTERVAL = 500  # mirrors kairos_papertrade.py's _PREWARM_GC_INTERVAL

# --gpu-workers auto sizing (see decide_gpu_workers() below / --gpu-workers-max
# help). per-worker estimates are seeds -- _phase1_with_vram_calibration()
# refines them chunk-to-chunk from what was actually observed.
VRAM_HEADROOM_MIB = 512
RAM_HEADROOM_MIB = 1024
DEFAULT_PER_WORKER_VRAM_MIB = 2200.0  # Kronos-small ~2.2GB/4-asset group; base ~3.6GB -- conservative seed
DEFAULT_PER_WORKER_RSS_MIB = 1100.0

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

def decide_gpu_workers(free_vram_mib, avail_ram_mib, per_worker_vram_mib,
                        per_worker_rss_mib, max_workers, cpu_workers_running=0):
    """Pure sizing rule for --gpu-workers auto. No I/O -- every input is a
    plain number, so this is unit-testable without a GPU. cpu_workers_running
    is the number of --workers phase-2 processes already competing for RAM
    (nonzero only under --pipeline, where phase 1 and phase 2 run at once)."""
    vram_allows = int((free_vram_mib - VRAM_HEADROOM_MIB) // per_worker_vram_mib)
    ram_for_workers = avail_ram_mib - RAM_HEADROOM_MIB - cpu_workers_running * per_worker_rss_mib
    ram_allows = int(ram_for_workers // per_worker_rss_mib)
    return max(1, min(vram_allows, ram_allows, max_workers))


def _probe_free_vram_mib():
    """Free VRAM in MiB via torch.cuda.mem_get_info() if torch+CUDA are
    available, else `nvidia-smi --query-gpu=memory.free`. Returns None (not
    0) when neither works -- no GPU on this box, CPU-only CI, etc -- so
    callers can tell "no GPU" apart from "GPU reports 0 MiB free"."""
    try:
        import torch
        if torch.cuda.is_available():
            free_b, _total_b = torch.cuda.mem_get_info()
            return free_b / (1024 ** 2)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


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
    """ponytail: coarse ~1s-poll min-value tracker used to self-calibrate the
    per-worker VRAM/RAM estimates chunk-to-chunk. Not an NVML event hook, and
    a concurrent process on the same GPU/box pollutes the reading -- good
    enough to nudge the next chunk's --gpu-workers auto decision, not a
    precise profiler. Upgrade path: NVML polling thread if this ever proves
    too noisy in practice."""

    def __init__(self, probe_fn, interval=1.0):
        self._probe_fn = probe_fn
        self._interval = interval
        self.min_value = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            v = self._probe_fn()
            if v is not None and (self.min_value is None or v < self.min_value):
                self.min_value = v
            self._stop.wait(self._interval)

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def _phase1_with_vram_calibration(gpu_workers, per_worker_vram_mib, per_worker_rss_mib,
                                   auto, probe_vram_fn, run_phase1_fn):
    """Runs run_phase1_fn() (a zero-arg closure around run_phase1()) and, only
    when auto, samples free VRAM/RAM around the call to refine
    per_worker_vram_mib/per_worker_rss_mib from what THIS phase actually
    used -- so the NEXT chunk's decide_gpu_workers() call sizes better than
    the fixed seed. No sampling overhead when auto is off.

    Returns (run_phase1 result, new_per_worker_vram_mib, new_per_worker_rss_mib).
    """
    if not auto:
        return run_phase1_fn(), per_worker_vram_mib, per_worker_rss_mib

    free_vram_before = probe_vram_fn()
    ram_before_mib = kp_predcache._read_mem_available_bytes() / (1024 ** 2)
    vram_sampler = _MinSampler(probe_vram_fn)
    ram_sampler = _MinSampler(lambda: kp_predcache._read_mem_available_bytes() / (1024 ** 2))
    with vram_sampler, ram_sampler:
        result = run_phase1_fn()

    if free_vram_before is not None and vram_sampler.min_value is not None and gpu_workers > 0:
        used = free_vram_before - vram_sampler.min_value
        if used > 0:
            per_worker_vram_mib = max(used / gpu_workers, 256.0)
    if ram_sampler.min_value is not None and gpu_workers > 0:
        used = ram_before_mib - ram_sampler.min_value
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


def run_phase1(chunk, model_path, interval, backtest_period, pred_samples,
               cache_dir, cache_max_bytes, gpu_workers, force_pool=False):
    """chunk: list of (group_id, assets, assets_key, sharpe). Returns (hits, misses, failures).

    force_pool=True always dispatches through ProcessPoolExecutor, even for
    gpu_workers<=1 -- required under --pipeline so phase 1's Python work runs
    in a worker process, not inline on the thread that's supposed to let
    phase 2 run concurrently (an inline call would hold the GIL on the main
    thread's ThreadPoolExecutor worker and block phase 2's own submits)."""
    total_hits = total_misses = 0
    failures = []
    if gpu_workers <= 1 and not force_pool:
        # Fully serial (default): no subprocess/pickling overhead, one GPU
        # user at a time -- the safe default for a single GPU box.
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
            for fut in as_completed(futures):
                _gid, ak, status, hits, misses, err = fut.result()
                if status == "fail":
                    failures.append((ak, err))
                else:
                    total_hits += hits
                    total_misses += misses
    return total_hits, total_misses, failures


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
    correlation_run_id = latest_correlation_run_id(conn, INTERVAL)
    groups = kp.select_deduped_groups(conn, correlation_run_id)
    prioritized = select_prioritized_groups(
        conn, groups, args.stage, model_path, args.require_stage, args.require_since,
    )
    conn.close()
    if args.limit is not None:
        prioritized = prioritized[: args.limit]

    print(f"Unprocessed groups: {len(prioritized)} (of {len(groups)} total deduped, "
          f"correlation run_id={correlation_run_id}). Stage={args.stage!r} model_path={model_path!r}. "
          f"Budget: {args.max_hours}h. Chunk size: {args.chunk_size}. Cache dir: {cache_dir}")

    chunks = list(_chunked(prioritized, args.chunk_size))
    done_total = stage_failed_total = prewarm_failed_total = 0
    per_worker_vram_mib = DEFAULT_PER_WORKER_VRAM_MIB
    per_worker_rss_mib = DEFAULT_PER_WORKER_RSS_MIB
    gpu_auto = args.gpu_workers == "auto"
    # Under --pipeline, phase 1's prewarm runs concurrently with phase 2's
    # --workers pool, so the RAM term of --gpu-workers auto's sizing must
    # treat those workers as already-committed RAM, not free RAM.
    cpu_workers_running = args.workers if args.pipeline else 0

    def _resolve_gw():
        return _resolve_gpu_workers_for_chunk(
            args.gpu_workers, args.gpu_workers_max, per_worker_vram_mib, per_worker_rss_mib,
            cpu_workers_running,
        )

    if args.pipeline:
        cpu_count = os.cpu_count() or 1
        physical = cpu_count // 2
        # auto mode's actual gpu-workers varies per chunk; use the ceiling for this startup check.
        worst_case_gw = args.gpu_workers_max if gpu_auto else args.gpu_workers
        if worst_case_gw + args.workers > physical:
            print(f"WARNING: --pipeline with gpu-workers(~{worst_case_gw}) + --workers {args.workers} "
                  f"= {worst_case_gw + args.workers} processes > {physical} physical cores "
                  f"(cpu_count={cpu_count} logical). Peak RAM may be tight (~1.1GB/prewarm process "
                  f"measured) -- consider a lower --workers.")

        def _phase1_call(chunk_):
            nonlocal per_worker_vram_mib, per_worker_rss_mib
            gw, gw_log = _resolve_gw()
            print(f"  [prewarm] groups={len(chunk_)} {gw_log}")
            result, per_worker_vram_mib, per_worker_rss_mib = _phase1_with_vram_calibration(
                gw, per_worker_vram_mib, per_worker_rss_mib, gpu_auto, _probe_free_vram_mib,
                lambda: run_phase1(chunk_, model_path, INTERVAL, BACKTEST_PERIOD, PRED_SAMPLES,
                                    cache_dir, cache_max_bytes, gw, force_pool=True),
            )
            return result

        def _phase2_call(chunk_):
            return run_phase2(chunk_, args.stage, model_path, INTERVAL, BACKTEST_PERIOD, PRED_SAMPLES,
                               extra_env, args.workers)

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
                    remaining = sum(len(c) for c in chunks[ci:])
                    print(f"\n[budget] {args.max_hours}h elapsed after chunk {ci}/{len(chunks)}. "
                          f"{remaining} groups remain for next invocation.")
                    break
        finally:
            gen.close()  # waits out any in-flight prefetch before returning
    else:
        for ci, chunk in enumerate(chunks, 1):
            if time.time() >= deadline:
                remaining = sum(len(c) for c in chunks[ci - 1:])
                print(f"\n[budget] {args.max_hours}h elapsed, stopping before chunk {ci}/{len(chunks)}. "
                      f"{remaining} groups remain for next invocation.")
                break

            gw, gw_log = _resolve_gw()
            if gpu_auto:
                print(f"[chunk {ci}/{len(chunks)}] {gw_log}")

            t0 = time.time()
            (hits, misses, p1_failures), per_worker_vram_mib, per_worker_rss_mib = _phase1_with_vram_calibration(
                gw, per_worker_vram_mib, per_worker_rss_mib, gpu_auto, _probe_free_vram_mib,
                lambda: run_phase1(chunk, model_path, INTERVAL, BACKTEST_PERIOD, PRED_SAMPLES,
                                    cache_dir, cache_max_bytes, gw),
            )
            t1 = time.time()
            # A prewarm failure for a group doesn't block phase 2 -- that group's
            # stage subprocess just falls back to loading the model itself.
            done, stage_failed, p2_failures = run_phase2(
                chunk, args.stage, model_path, INTERVAL, BACKTEST_PERIOD, PRED_SAMPLES,
                extra_env, args.workers,
            )
            t2 = time.time()

            done_total += done
            stage_failed_total += stage_failed
            prewarm_failed_total += len(p1_failures)

            for ak, err in p1_failures:
                print(f"  [prewarm FAIL] {ak}: {err}")
            for ak, err in p2_failures:
                print(f"  [stage FAIL] {ak}: {err}")

            print(f"[chunk {ci}/{len(chunks)}] groups={len(chunk)} prewarm_hits={hits} "
                  f"prewarm_misses={misses} phase1_s={t1 - t0:.1f} phase2_s={t2 - t1:.1f} "
                  f"done_total={done_total} stage_failed_total={stage_failed_total} "
                  f"prewarm_failed_total={prewarm_failed_total}")

    print(f"\nParallel {args.stage} sweep done: {done_total} run, {stage_failed_total} stage failures, "
          f"{prewarm_failed_total} prewarm failures, total={len(prioritized)}.")


if __name__ == "__main__":
    main()
