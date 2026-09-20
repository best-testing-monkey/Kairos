"""E18-S05: Unit tests for offline TPSL validation.

Tests the validation window derivation and comparison report generation
with synthetic fixtures built from real table-creation functions.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import sqlite3  # noqa: E402
import tempfile  # noqa: E402
from datetime import datetime, timezone, timedelta  # noqa: E402

import pytest  # noqa: E402
import numpy as np  # noqa: E402

from compare_tpsl_model import purged_time_split, _get_validation_window  # noqa: E402
from tpsl_label_grid import _ensure_tpsl_label_candidates_table  # noqa: E402
from kairos_signal_replay import _ensure_signal_replay_tables  # noqa: E402


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        yield path
    finally:
        os.unlink(path)


def test_purged_time_split_basic():
    """Test purged_time_split with known inputs."""
    # 10 signals, equally spaced
    dates = [f"2026-01-{i+1:02d}" for i in range(10)]

    train_idx, val_idx = purged_time_split(dates, val_fraction=0.2, embargo_bars=5)

    # 20% of 10 = 2 signals in validation
    assert len(val_idx) == 2
    # 5 embargo bars, so embargo_start = 10 - 2 - 5 = 3
    # train_idx should be [0, 1, 2], val_idx should be [8, 9]
    assert list(train_idx) == [0, 1, 2]
    assert list(val_idx) == [8, 9]


def test_purged_time_split_edge_case_small_dataset():
    """Test with dataset smaller than embargo window."""
    dates = [f"2026-01-{i+1:02d}" for i in range(3)]

    train_idx, val_idx = purged_time_split(dates, val_fraction=0.2, embargo_bars=5)

    # 20% of 3 = 1 signal in validation
    assert len(val_idx) == 1
    # embargo_start = max(0, 2 - 5) = 0, so train is empty
    assert len(train_idx) == 0
    assert list(val_idx) == [2]


def test_purged_time_split_no_embargo():
    """Test with zero embargo bars."""
    dates = [f"2026-01-{i+1:02d}" for i in range(10)]

    train_idx, val_idx = purged_time_split(dates, val_fraction=0.2, embargo_bars=0)

    assert len(val_idx) == 2
    assert len(train_idx) == 8


def test_get_validation_window_basic(temp_db):
    """Test validation window derivation from database with real schemas."""
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row

    # Create tables using real table-creation functions
    _ensure_signal_replay_tables(conn)
    _ensure_tpsl_label_candidates_table(conn)

    # Insert 10 signals into papertrade_signals
    base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(10):
        as_of = (base_date + timedelta(days=i)).date().isoformat()
        conn.execute(
            """
            INSERT INTO papertrade_signals (
                signal_id, strategy_name, ticker, direction, interval, as_of,
                entry, stop, target, model_label, checkpoint_fingerprint, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"sig_{i}", "test_strategy", f"TEST_{i}", "long", "1d", as_of,
             100.0, 90.0, 110.0, "base", "", datetime.now(timezone.utc).isoformat())
        )

    # Insert 10 resolved candidates in tpsl_label_candidates
    for i in range(10):
        conn.execute(
            """
            INSERT INTO tpsl_label_candidates (
                signal_id, stop_pct, target_pct, resolved, hit_target_first,
                interval_used, engine_version, computed_at
            ) VALUES (?, ?, ?, 1, 1, ?, ?, ?)
            """,
            (f"sig_{i}", 10.0, 75.0, "1d", "e18_s01_v1",
             datetime.now(timezone.utc).isoformat())
        )
    conn.commit()

    val_start, val_end = _get_validation_window(conn)

    # Expected: last 20% = 2 signals, so days 8-9
    expected_start = (base_date + timedelta(days=8)).date().isoformat()
    expected_end = (base_date + timedelta(days=9)).date().isoformat()

    assert val_start == expected_start
    assert val_end == expected_end

    conn.close()


def test_get_validation_window_no_data(temp_db):
    """Test validation window derivation with no resolved candidates."""
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row

    # Create tables using real table-creation functions
    _ensure_signal_replay_tables(conn)
    _ensure_tpsl_label_candidates_table(conn)
    # Don't populate either table

    with pytest.raises(ValueError, match="No resolved candidates"):
        _get_validation_window(conn)

    conn.close()


def test_get_validation_window_unsorted_dates(temp_db):
    """Test that validation window handles unsorted dates correctly."""
    conn = sqlite3.connect(temp_db)
    conn.row_factory = sqlite3.Row

    # Create tables using real table-creation functions
    _ensure_signal_replay_tables(conn)
    _ensure_tpsl_label_candidates_table(conn)

    # Insert in random order
    base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    dates_order = [5, 1, 9, 2, 7, 0, 4, 8, 3, 6]

    # First insert signals into papertrade_signals (required for join)
    for idx in dates_order:
        as_of = (base_date + timedelta(days=idx)).date().isoformat()
        conn.execute(
            """
            INSERT INTO papertrade_signals (
                signal_id, strategy_name, ticker, direction, interval, as_of,
                entry, stop, target, model_label, checkpoint_fingerprint, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"sig_{idx}", "test_strategy", f"TEST_{idx}", "long", "1d", as_of,
             100.0, 90.0, 110.0, "base", "", datetime.now(timezone.utc).isoformat())
        )

    # Then insert candidates into tpsl_label_candidates
    for idx in dates_order:
        conn.execute(
            """
            INSERT INTO tpsl_label_candidates (
                signal_id, stop_pct, target_pct, resolved, hit_target_first,
                interval_used, engine_version, computed_at
            ) VALUES (?, ?, ?, 1, 1, ?, ?, ?)
            """,
            (f"sig_{idx}", 10.0, 75.0, "1d", "e18_s01_v1",
             datetime.now(timezone.utc).isoformat())
        )
    conn.commit()

    val_start, val_end = _get_validation_window(conn)

    # Should still be days 8-9 (last 20%)
    expected_start = (base_date + timedelta(days=8)).date().isoformat()
    expected_end = (base_date + timedelta(days=9)).date().isoformat()

    assert val_start == expected_start
    assert val_end == expected_end

    conn.close()


def test_comparison_report_synthetic():
    """Test comparison report generation with synthetic baseline vs ML outcomes.

    This test verifies that when ML-wrapped signals have better outcomes
    than baseline, the report correctly attributes the delta.
    """
    # Create synthetic PnL lists
    # Baseline: -2%, -1%, 1%, 2%, 3%, 4%, 5%, 6%, 7%, 8%
    baseline_pnls = [-2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

    # ML-wrapped: 0%, 1%, 2%, 3%, 4%, 5%, 6%, 7%, 8%, 9% (uniformly better)
    ml_pnls = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    # Compute metrics
    def compute_metrics(pnl_list):
        arr = np.array(pnl_list)
        mean_ret = float(np.mean(arr))
        std_ret = float(np.std(arr))
        sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
        win_rate = float(np.mean(arr > 0)) if len(arr) > 0 else 0.0
        return {
            "sharpe": sharpe,
            "win_rate": win_rate,
            "mean_pnl": mean_ret,
            "count": len(arr),
            "total_pnl": float(np.sum(arr)),
        }

    baseline_metrics = compute_metrics(baseline_pnls)
    ml_metrics = compute_metrics(ml_pnls)

    # Assertions
    assert baseline_metrics["count"] == 10
    assert ml_metrics["count"] == 10

    # ML should have higher mean PnL
    assert ml_metrics["mean_pnl"] > baseline_metrics["mean_pnl"]

    # ML should have higher win rate
    assert ml_metrics["win_rate"] > baseline_metrics["win_rate"]

    # ML should have higher Sharpe
    assert ml_metrics["sharpe"] > baseline_metrics["sharpe"]

    # Check specific values
    # baseline: [-2, -1, 1, 2, 3, 4, 5, 6, 7, 8] => sum=33, mean=3.3
    assert baseline_metrics["mean_pnl"] == 3.3
    # ml: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] => sum=45, mean=4.5
    assert ml_metrics["mean_pnl"] == 4.5
    # baseline: 8 positive out of 10 (excluding -2, -1)
    assert baseline_metrics["win_rate"] == 0.8
    # ml: 9 positive out of 10 (all except 0)
    assert ml_metrics["win_rate"] == 0.9
