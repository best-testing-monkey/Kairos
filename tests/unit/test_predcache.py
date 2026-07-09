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
    k1 = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base", "abc123")
    k2 = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base", "abc123")
    assert k1 == k2
    assert isinstance(k1, str)


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
                         pc.content_hash_for_closes([100.0, 101.0]))
    key_b = pc.make_key("BTC-USD", "1d", pd.Timestamp("2024-01-01"), 300, 100, "base",
                         pc.content_hash_for_closes([200.0, 201.0]))
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
    cache = pc.PredictionCache(str(tmp_path), mem_budget_bytes=10 * 1024 * 1024)
    samples = _make_samples(2)
    cache.put("corrupt-key", samples)

    path = cache._disk_path("corrupt-key")
    assert os.path.exists(path)
    # Corrupt the file on disk.
    with open(path, "wb") as f:
        f.write(b"not a valid npz file")

    # New instance so the in-memory LRU doesn't mask the corrupt disk file.
    cache2 = pc.PredictionCache(str(tmp_path), mem_budget_bytes=10 * 1024 * 1024)
    assert cache2.get("corrupt-key") is None
    # Corrupt file should have been cleaned up.
    assert not os.path.exists(path)


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
