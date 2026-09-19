"""E18-S03: Tests for scripts/train_tpsl_model.py.

Tests cover:
- purged_time_split(): documented embargo gap, no adjacent-timestamp leakage
- check_class_sufficiency(): per-class fallback-to-pooled decision
- GradientBoostedStumps pickle round-trip: identical predict_proba before/after
- Full train_and_save() pipeline wiring against a synthetic sqlite DB
"""
import json
import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from kairos_ml import GradientBoostedStumps  # noqa: E402
from kairos_signal_replay import _ensure_signal_replay_tables  # noqa: E402

import train_tpsl_model as train_module  # noqa: E402
from train_tpsl_model import (  # noqa: E402
    check_class_sufficiency,
    purged_time_split,
    train_and_save,
)


class TestPurgedTimeSplit:
    """purged_time_split(): documented default is val_fraction=0.2, embargo_bars=5."""

    def test_embargo_gap_matches_documented_default(self):
        """100 sorted daily signals: 20 held out as val, 5-row embargo gap before it."""
        as_of = pd.date_range("2024-01-01", periods=100, freq="D")

        train_idx, val_idx = purged_time_split(as_of, val_fraction=0.2, embargo_bars=5)

        assert len(val_idx) == 20
        assert len(train_idx) == 100 - 20 - 5
        # Positions are contiguous blocks in the (already time-sorted) input:
        # train = [0, 75), embargo = [75, 80) dropped, val = [80, 100).
        assert val_idx.min() - train_idx.max() == 5 + 1
        # Embargoed rows are in neither set.
        assert set(train_idx.tolist()) & set(val_idx.tolist()) == set()

    def test_sorts_unsorted_input_before_splitting(self):
        """Function sorts by as_of itself -- shuffled input must not leak adjacency."""
        n = 50
        as_of = pd.date_range("2024-01-01", periods=n, freq="D")
        perm = np.random.default_rng(0).permutation(n)
        shuffled = as_of[perm]

        train_idx, val_idx = purged_time_split(shuffled, val_fraction=0.2, embargo_bars=3)

        train_times = shuffled[train_idx]
        val_times = shuffled[val_idx]
        # Every train timestamp must be strictly earlier than every val timestamp,
        # regardless of the shuffled input order.
        assert train_times.max() < val_times.min()

    def test_no_crash_on_small_n(self):
        """Degenerate small-n case still returns usable, non-overlapping index arrays."""
        as_of = pd.date_range("2024-01-01", periods=5, freq="D")

        train_idx, val_idx = purged_time_split(as_of, val_fraction=0.2, embargo_bars=5)

        assert len(val_idx) >= 1
        assert set(train_idx.tolist()) & set(val_idx.tolist()) == set()


class TestCheckClassSufficiency:
    """check_class_sufficiency(): falls back to pooled when any class (or the
    whole set) is below CLASS_STATS_MIN_SIGNALS -- mirrors kairos_pipeline's
    strategy_class_stats() fallback-to-corpus precedent."""

    def test_falls_back_when_one_class_below_threshold(self):
        classes = ["equity"] * 40 + ["crypto"] * 5

        ok, counts = check_class_sufficiency(classes, min_signals=30)

        assert ok is False
        assert counts == {"equity": 40, "crypto": 5}

    def test_per_class_ok_when_every_class_sufficient(self):
        classes = ["equity"] * 40 + ["crypto"] * 35

        ok, counts = check_class_sufficiency(classes, min_signals=30)

        assert ok is True
        assert counts == {"equity": 40, "crypto": 35}

    def test_falls_back_when_whole_set_below_threshold(self):
        """Even a single class fails the check if the total is too small."""
        classes = ["equity"] * 10

        ok, counts = check_class_sufficiency(classes, min_signals=30)

        assert ok is False
        assert counts == {"equity": 10}


class TestGradientBoostedStumpsPickleRoundTrip:
    """Saved classifier round-trips through pickle with identical predict_proba()."""

    def test_predict_proba_identical_after_pickle(self, tmp_path):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(float)

        model = GradientBoostedStumps(n_trees=10, lr=0.1, seed=1)
        model.fit(X, y)
        before = model.predict_proba(X)

        pkl_path = tmp_path / "model.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump(model, fh)
        with open(pkl_path, "rb") as fh:
            loaded = pickle.load(fh)

        after = loaded.predict_proba(X)
        assert np.array_equal(before, after)


def _build_synthetic_db(tmp_path):
    """DB with 50 resolved signals x the standard 3x3 stop/target grid."""
    db_path = str(tmp_path / "pipeline_results.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _ensure_signal_replay_tables(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tpsl_label_candidates (
            signal_id       TEXT NOT NULL,
            stop_pct        REAL NOT NULL,
            target_pct      REAL NOT NULL,
            resolved        INTEGER NOT NULL,
            hit_target_first INTEGER,
            interval_used   TEXT,
            engine_version  TEXT NOT NULL,
            computed_at     TEXT NOT NULL,
            PRIMARY KEY (signal_id, stop_pct, target_pct)
        )
        """
    )
    conn.commit()

    stop_grid = [10.0, 15.0, 20.0]
    target_grid = [75.0, 85.0, 90.0]
    rng = np.random.default_rng(3)

    for i in range(50):
        signal_id = f"sig{i:03d}"
        as_of = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)).isoformat()
        conn.execute(
            "INSERT INTO papertrade_signals (signal_id, strategy_name, ticker, direction, "
            "interval, as_of, entry, stop, target, model_label, checkpoint_fingerprint, created_at) "
            "VALUES (?, 'test', 'AAPL', 'long', '1d', ?, 100.0, 90.0, 175.0, 'test', '', ?)",
            (signal_id, as_of, as_of),
        )
        for stop_pct in stop_grid:
            for target_pct in target_grid:
                hit = int(rng.integers(0, 2))
                conn.execute(
                    "INSERT INTO tpsl_label_candidates (signal_id, stop_pct, target_pct, resolved, "
                    "hit_target_first, interval_used, engine_version, computed_at) "
                    "VALUES (?, ?, ?, 1, ?, '1d', 'test', ?)",
                    (signal_id, stop_pct, target_pct, hit, as_of),
                )
    conn.commit()
    return db_path, conn


class TestTrainAndSavePipeline:
    """End-to-end wiring: DB join -> feature extraction -> 9 saved models + metadata.

    price_cache/history fetching itself is E18-S02's concern (test_tpsl_features.py);
    _fetch_history is monkeypatched here so this test is purely about the
    join/train/save wiring in train_and_save().
    """

    def test_writes_one_pooled_model_per_grid_candidate(self, tmp_path, monkeypatch):
        db_path, conn = _build_synthetic_db(tmp_path)

        # 60 bars of synthetic OHLCV, entirely before every signal's as_of (Jan 2024+),
        # so extract_features' internal as_of-truncation is a no-op here and every
        # signal gets the full, sufficient (>=21 bar) history regardless of its date.
        idx = pd.date_range("2023-10-01", periods=60, freq="D")
        closes = 100.0 + np.cumsum(np.random.default_rng(1).normal(size=60))
        canned_history = pd.DataFrame(
            {"open": closes, "high": closes * 1.01, "low": closes * 0.99,
             "close": closes, "volume": 1000.0},
            index=idx,
        )
        monkeypatch.setattr(train_module, "_fetch_history", lambda *a, **k: canned_history)

        output_dir = str(tmp_path / "tpsl_models")
        try:
            # min_signals=1000 forces the pooled fallback deterministically --
            # matches the real corpus's expected behavior (equity-skewed, per
            # this repo's tracked project state) and pins the filename shape
            # the ticket describes (<stop_pct>_<target_pct>.pkl).
            summary = train_and_save(conn, db_path, output_dir, min_signals=1000)

            assert summary["trained"] == 9
            pkl_files = sorted(f for f in os.listdir(output_dir) if f.endswith(".pkl"))
            assert len(pkl_files) == 9

            with open(os.path.join(output_dir, "feature_metadata.json")) as fh:
                meta = json.load(fh)
            assert len(meta["feature_columns"]) > 0
            assert len(meta["candidates"]) == 9
            assert all(v["mode"] == "pooled" for v in meta["candidates"].values())

            with open(os.path.join(output_dir, pkl_files[0]), "rb") as fh:
                loaded = pickle.load(fh)
            assert loaded.fitted
            assert isinstance(loaded, GradientBoostedStumps)
        finally:
            conn.close()

    def test_no_table_returns_gracefully(self, tmp_path):
        """No tpsl_label_candidates table yet (E18-S01 not run) -- no crash."""
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        _ensure_signal_replay_tables(conn)  # papertrade_signals only, no tpsl_label_candidates
        try:
            summary = train_and_save(conn, db_path, str(tmp_path / "out"))
            assert summary["trained"] == 0
        finally:
            conn.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
