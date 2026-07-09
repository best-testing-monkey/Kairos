"""
Per-run prediction cache shared across kairos_pipeline.py subprocesses.

Motivation: --stage auto now builds overlapping correlation groups, so the
same symbol can appear in several groups within one auto run. Each group is
backtested in its own subprocess (kairos_strategies.py via
run_backtest_subprocess), so identical per-bar Kronos predictions would
otherwise be recomputed once per group.

This module provides a two-layer cache:
  - Disk layer (one .npz file per key) so predictions survive across
    subprocess boundaries within the same run.
  - An in-memory LRU on top, bounded by a byte budget derived from
    /proc/meminfo's MemAvailable (stdlib only, no psutil), to avoid holding
    an unbounded amount of decoded DataFrame data in RAM within one process.

The cache is opt-in: it only activates when KAIROS_PRED_CACHE_DIR is set in
the environment (kairos_pipeline.py sets this for the lifetime of a --stage
auto run and removes the directory afterwards). Single-stage invocations
never set the variable, so behavior is byte-identical to before this module
existed.
"""

import hashlib
import os
import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import pandas as pd

# Columns persisted for each prediction sample DataFrame.
_DEFAULT_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]


def content_hash_for_closes(closes) -> str:
    """Cheap content hash for a lookback window's close column.

    Used as part of the cache key so that a stale or different input window
    (e.g. a later run with more/different history) never collides with an
    earlier cached prediction for the "same" (symbol, interval, bar) key.
    """
    arr = np.asarray(closes, dtype=np.float64)
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def make_key(symbol, interval, bar_timestamp, lookback_len, pred_samples,
             model_id, content_hash) -> str:
    """Build a canonical, filename-safe cache key string."""
    bar_ts_iso = pd.Timestamp(bar_timestamp).isoformat()
    parts = [
        str(symbol), str(interval), bar_ts_iso, str(lookback_len),
        str(pred_samples), str(model_id or "default"), str(content_hash),
    ]
    return "|".join(parts)


def _filename_for_key(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return f"{digest}.npz"


def _read_mem_available_bytes() -> int:
    """Read MemAvailable from /proc/meminfo, in bytes. Falls back to a
    conservative default if unavailable (e.g. non-Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 512 * 1024 * 1024  # 512MB fallback


def _dfs_nbytes(sample_dfs: List[pd.DataFrame]) -> int:
    total = 0
    for df in sample_dfs:
        total += df.to_numpy(dtype=np.float64, copy=False).nbytes
    return total


class PredictionCache:
    """Disk-backed prediction cache with an in-memory LRU on top.

    get(key) -> Optional[List[pd.DataFrame]]
    put(key, sample_dfs)
    """

    def __init__(self, cache_dir, mem_fraction: float = 0.25, mem_budget_bytes: Optional[int] = None):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        if mem_budget_bytes is not None:
            self.mem_budget_bytes = int(mem_budget_bytes)
        else:
            self.mem_budget_bytes = int(_read_mem_available_bytes() * mem_fraction)
        self._lock = threading.Lock()
        self._mem: "OrderedDict[str, tuple]" = OrderedDict()  # key -> (list[DataFrame], nbytes)
        self._mem_bytes = 0

    # ── in-memory LRU ────────────────────────────────────────────────────
    def _mem_get(self, key: str):
        with self._lock:
            entry = self._mem.get(key)
            if entry is None:
                return None
            self._mem.move_to_end(key)
            return entry[0]

    def _mem_put(self, key: str, sample_dfs: List[pd.DataFrame]):
        nbytes = _dfs_nbytes(sample_dfs)
        with self._lock:
            if key in self._mem:
                _, old_nbytes = self._mem.pop(key)
                self._mem_bytes -= old_nbytes
            self._mem[key] = (sample_dfs, nbytes)
            self._mem_bytes += nbytes
            while self._mem_bytes > self.mem_budget_bytes and len(self._mem) > 0:
                evict_key, (_, evict_nbytes) = self._mem.popitem(last=False)
                if evict_key == key:
                    # Don't evict the entry we just inserted if it's the only
                    # one and still over budget alone; keep it to avoid a
                    # get-miss-immediately-after-put surprise.
                    self._mem[evict_key] = (self._mem.get(evict_key, (sample_dfs, evict_nbytes))[0], evict_nbytes)
                    self._mem_bytes += evict_nbytes
                    break
                self._mem_bytes -= evict_nbytes

    # ── disk layer ───────────────────────────────────────────────────────
    def _disk_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, _filename_for_key(key))

    def _disk_read(self, key: str) -> Optional[List[pd.DataFrame]]:
        path = self._disk_path(key)
        if not os.path.exists(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as npz:
                values = npz["values"]  # (n_samples, n_rows, n_cols) float32
                columns = list(npz["columns"])
                index_iso = npz["index"]  # (n_samples, n_rows) unicode strings
            columns = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in columns]
            n_samples = values.shape[0]
            dfs = []
            for i in range(n_samples):
                idx = pd.to_datetime([
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in index_iso[i]
                ])
                dfs.append(pd.DataFrame(values[i].astype(np.float64), columns=columns, index=idx))
            return dfs
        except Exception:
            # Corrupt/unreadable file: treat as a miss and clean it up.
            try:
                os.remove(path)
            except OSError:
                pass
            return None

    def _disk_write(self, key: str, sample_dfs: List[pd.DataFrame]):
        if not sample_dfs:
            return
        columns = list(sample_dfs[0].columns)
        n_rows = len(sample_dfs[0])
        # All samples share the same shape in practice (pred_len fixed per call).
        values = np.stack([
            df[columns].to_numpy(dtype=np.float32) for df in sample_dfs
        ], axis=0)
        index_iso = np.array([
            [pd.Timestamp(ts).isoformat() for ts in df.index]
            for df in sample_dfs
        ])
        path = self._disk_path(key)
        tmp_path = path + f".tmp{os.getpid()}"
        with open(tmp_path, "wb") as f:
            np.savez(f, values=values, columns=np.array(columns), index=index_iso)
        os.replace(tmp_path, path)

    # ── public API ───────────────────────────────────────────────────────
    def get(self, key: str) -> Optional[List[pd.DataFrame]]:
        cached = self._mem_get(key)
        if cached is not None:
            return cached
        from_disk = self._disk_read(key)
        if from_disk is not None:
            self._mem_put(key, from_disk)
        return from_disk

    def put(self, key: str, sample_dfs: List[pd.DataFrame]):
        self._mem_put(key, sample_dfs)
        self._disk_write(key, sample_dfs)


_singleton: Optional[PredictionCache] = None
_singleton_dir: Optional[str] = None


def get_cache() -> Optional[PredictionCache]:
    """Return the process-wide PredictionCache singleton, or None if the
    cache is disabled (KAIROS_PRED_CACHE_DIR unset)."""
    global _singleton, _singleton_dir
    cache_dir = os.environ.get("KAIROS_PRED_CACHE_DIR")
    if not cache_dir:
        return None
    if _singleton is not None and _singleton_dir == cache_dir:
        return _singleton
    _singleton = PredictionCache(cache_dir)
    _singleton_dir = cache_dir
    return _singleton
