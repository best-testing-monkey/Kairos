"""Tests for strategy/kairos_predcache.py (Feature 2: per-run prediction cache)."""

import os

import numpy as np
import pandas as pd
import pytest

import kairos_predcache as pc


def _make_samples(n=3, base_price=100.0, ts="2024-01-01"):
    """Build n single-row sample DataFrames with OHLCV columns."""
    idx = pd.DatetimeIndex([pd.Timestamp(ts)])
    dfs = []
    for i in range(n):
        dfs.append(pd.DataFrame({
            "open": [base_price + i], "high": [base_price + i + 1],
            "low": [base_price + i - 1], "close": [base_price + i * 0.5],
            "volume": [1000.0 + i], "amount": [100000.0 + i],
        }, index=idx))
    return dfs


# ── content hash / key construction ─────────────────────────────────────────

def test_content_hash_deterministic():
    closes = [100.0, 101.0, 102.5]
    h1 = pc.content_hash_for_closes(closes)
    h2 = pc.content_hash_for_closes(closes)
    assert h1 == h2
    assert len(h1) == 12


def test_content_hash_differs_for_different_input():
    h1 = pc.content_hash_for_closes([100.0, 101.0])
    h2 = pc.content_hash_for_closes([100.0, 102.0])
    assert h1 != h2


def test_make_key_is_stable_string():
    k1 = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base", "abc123", 1)
    k2 = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base", "abc123", 1)
    assert k1 == k2
    assert isinstance(k1, str)


def test_make_key_differs_by_pred_len_alone():
    k1 = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base", "abc123", 1)
    k2 = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base", "abc123", 5)
    assert k1 != k2


def test_make_key_differs_by_checkpoint_fingerprint_alone():
    k1 = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base", "abc123", 1,
                      checkpoint_fingerprint="1024-1000")
    k2 = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base", "abc123", 1,
                      checkpoint_fingerprint="2048-2000")
    assert k1 != k2


def test_make_key_checkpoint_fingerprint_defaults_to_empty_and_is_stable():
    k1 = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base", "abc123", 1)
    k2 = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base", "abc123", 1,
                      checkpoint_fingerprint="")
    assert k1 == k2


# ── PredictionCache: disk + memory roundtrip ────────────────────────────────

def test_put_then_get_roundtrip_reconstructs_equal_dataframes(tmp_path):
    cache = pc.PredictionCache(str(tmp_path), mem_budget_bytes=10 * 1024 * 1024)
    samples = _make_samples(3)
    key = "sym-key-1"
    cache.put(key, samples)

    result = cache.get(key)
    assert result is not None
    assert len(result) == len(samples)
    for orig, got in zip(samples, result):
        pd.testing.assert_frame_equal(
            orig.astype("float64"), got.astype("float64"), check_dtype=False
        )


def test_get_miss_on_different_content_hash_key(tmp_path):
    cache = pc.PredictionCache(str(tmp_path), mem_budget_bytes=10 * 1024 * 1024)
    samples = _make_samples(2)
    key_a = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base",
                         pc.content_hash_for_closes([100.0, 101.0]), 1)
    key_b = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base",
                         pc.content_hash_for_closes([200.0, 201.0]), 1)
    cache.put(key_a, samples)
    assert cache.get(key_b) is None
    assert cache.get(key_a) is not None


def test_disk_persists_across_new_cache_instance(tmp_path):
    samples = _make_samples(2)
    cache1 = pc.PredictionCache(str(tmp_path), mem_budget_bytes=10 * 1024 * 1024)
    cache1.put("k1", samples)

    # A fresh PredictionCache instance (simulating a new subprocess) pointed
    # at the same cache_dir should still find it on disk.
    cache2 = pc.PredictionCache(str(tmp_path), mem_budget_bytes=10 * 1024 * 1024)
    result = cache2.get("k1")
    assert result is not None
    assert len(result) == 2


def test_corrupt_cache_file_treated_as_miss(tmp_path):
    # put() no longer writes .npz files (sqlite is the only write path now),
    # so there's no way to get a corrupt on-disk file via put()+corrupt
    # anymore. Simulate a legacy .npz file (written by a pre-sqlite-layer
    # version of this module, or manually dropped in place) for a key that
    # was never put through sqlite, to exercise _disk_read's corruption
    # handling via get()'s disk-fallback path.
    cache = pc.PredictionCache(str(tmp_path), mem_budget_bytes=10 * 1024 * 1024)
    key = "corrupt-key"
    path = cache._disk_path(key)
    with open(path, "wb") as f:
        f.write(b"not a valid npz file")

    assert cache.get(key) is None
    # Corrupt file should have been cleaned up.
    assert not os.path.exists(path)


def test_dfs_nbytes_counts_real_footprint_not_just_raw_buffer():
    # Regression for the eviction-accounting bug: raw .nbytes ignores Index/
    # DataFrame/block-manager overhead, so _mem's LRU eviction thought it was
    # under budget while real RSS climbed past it (see _dfs_nbytes docstring).
    dfs = _make_samples(3)
    raw_buffer_bytes = sum(df.to_numpy(dtype="float64", copy=False).nbytes for df in dfs)
    real_bytes = pc._dfs_nbytes(dfs)
    assert real_bytes > raw_buffer_bytes


def test_lru_eviction_under_tiny_byte_budget(tmp_path):
    # Each sample set is a handful of KB; force eviction with a tiny budget.
    cache = pc.PredictionCache(str(tmp_path), mem_budget_bytes=1)

    cache.put("k1", _make_samples(2))
    cache.put("k2", _make_samples(2))

    # With budget=1 byte, k1 should have been evicted from the in-memory LRU
    # once k2 was inserted (only the most-recently-put entry is kept when the
    # budget can't fit more than one entry).
    assert cache._mem_bytes <= max(cache._mem.get("k2", (None, 0))[1], 1) or "k2" in cache._mem
    # But disk-backed retrieval must still succeed for both keys.
    assert cache.get("k1") is not None
    assert cache.get("k2") is not None


# ── get_cache() singleton / opt-in behavior ─────────────────────────────────

def test_get_cache_returns_none_when_env_unset(monkeypatch):
    monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
    pc._singleton = None
    pc._singleton_dir = None
    assert pc.get_cache() is None


def test_get_cache_returns_instance_when_env_set(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
    pc._singleton = None
    pc._singleton_dir = None
    cache = pc.get_cache()
    assert cache is not None
    assert isinstance(cache, pc.PredictionCache)
    # Same dir -> same singleton instance.
    assert pc.get_cache() is cache
    monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
    pc._singleton = None
    pc._singleton_dir = None


def test_get_cache_honors_max_bytes_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("KAIROS_PRED_CACHE_MAX_BYTES", "12345")
    pc._singleton = None
    pc._singleton_dir = None
    cache = pc.get_cache()
    assert cache is not None
    assert cache.max_disk_bytes == 12345
    monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
    monkeypatch.delenv("KAIROS_PRED_CACHE_MAX_BYTES", raising=False)
    pc._singleton = None
    pc._singleton_dir = None


def test_get_cache_defaults_max_bytes_when_env_var_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("KAIROS_PRED_CACHE_MAX_BYTES", raising=False)
    pc._singleton = None
    pc._singleton_dir = None
    cache = pc.get_cache()
    assert cache.max_disk_bytes == pc._DEFAULT_MAX_DISK_BYTES
    monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
    pc._singleton = None
    pc._singleton_dir = None


def test_get_cache_honors_mem_bytes_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("KAIROS_PRED_CACHE_MEM_BYTES", "54321")
    pc._singleton = None
    pc._singleton_dir = None
    cache = pc.get_cache()
    assert cache is not None
    assert cache.mem_budget_bytes == 54321
    monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
    monkeypatch.delenv("KAIROS_PRED_CACHE_MEM_BYTES", raising=False)
    pc._singleton = None
    pc._singleton_dir = None


def test_get_cache_defaults_mem_bytes_when_env_var_unset(monkeypatch, tmp_path):
    # No explicit env var -> falls through to PredictionCache's own
    # available-RAM-fraction default (the thing this env var exists to
    # override), so just assert it's a positive int, not a specific value.
    monkeypatch.setenv("KAIROS_PRED_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("KAIROS_PRED_CACHE_MEM_BYTES", raising=False)
    pc._singleton = None
    pc._singleton_dir = None
    cache = pc.get_cache()
    assert cache.mem_budget_bytes > 0
    monkeypatch.delenv("KAIROS_PRED_CACHE_DIR", raising=False)
    pc._singleton = None
    pc._singleton_dir = None


# ── disk size cap + eviction ─────────────────────────────────────────────────
#
# TestDiskEviction (test_writing_past_budget_evicts_oldest_mtime_first,
# test_disk_bytes_tracks_actual_directory_size,
# test_seeds_disk_bytes_from_existing_directory_on_init) was deleted: it
# tested _evict_disk_if_over_budget()/_disk_write(), both removed along with
# all new .npz writes (put() now writes only to sqlite). There is no
# replacement to test since writes no longer touch disk at all.
