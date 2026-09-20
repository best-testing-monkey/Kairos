"""Unit tests for tpsl_label_grid.py.

Tests cover:
- Synthetic price paths with known outcomes for multiple candidates
- Disqualification when bracket never resolves
- Idempotency (re-running produces zero new rows)
- price_cache configuration before lookups
"""
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from kairos_signal_replay import (  # noqa: E402
    _ensure_signal_replay_tables,
)


def _setup_test_db() -> tuple[str, sqlite3.Connection]:
    """Create a temporary test database with schema."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test_pipeline_results.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Create schema (papertrade_signals + tpsl_label_candidates)
    _ensure_signal_replay_tables(conn)

    # Create tpsl_label_candidates table
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

    return db_path, conn


def _insert_signal(
    conn,
    signal_id: str,
    ticker: str,
    direction: str,
    interval: str,
    as_of: str,
    entry: float,
    stop: float,
    target: float,
) -> None:
    """Insert a papertrade_signals row for testing."""
    conn.execute(
        """
        INSERT INTO papertrade_signals (
            signal_id, strategy_name, ticker, direction, interval,
            as_of, entry, stop, target, model_label, checkpoint_fingerprint,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal_id,
            "test_strategy",
            ticker,
            direction,
            interval,
            as_of,
            entry,
            stop,
            target,
            "test",
            "",
            datetime.now(timezone.utc).isoformat(),
        )
    )
    conn.commit()


def test_compute_candidate_prices_long():
    """Test stop/target price computation for LONG direction."""
    from scripts.tpsl_label_grid import _compute_candidate_prices

    entry = 100.0
    stop_pct = 10.0
    target_pct = 75.0

    stop, target = _compute_candidate_prices(entry, "long", stop_pct, target_pct)

    # For LONG: stop below entry, target above
    assert stop == pytest.approx(90.0)  # 100 * (1 - 0.10)
    assert target == pytest.approx(175.0)  # 100 * (1 + 0.75)


def test_compute_candidate_prices_short():
    """Test stop/target price computation for SHORT direction."""
    from scripts.tpsl_label_grid import _compute_candidate_prices

    entry = 100.0
    stop_pct = 10.0
    target_pct = 75.0

    stop, target = _compute_candidate_prices(entry, "short", stop_pct, target_pct)

    # For SHORT: stop above entry, target below
    assert stop == pytest.approx(110.0)  # 100 * (1 + 0.10)
    assert target == pytest.approx(25.0)  # 100 * (1 - 0.75)


class TestResolveCandidate:
    """Test candidate resolution against price paths."""

    def test_long_target_hit_first(self):
        """Long trade where target hits before stop."""
        from scripts.tpsl_label_grid import resolve_candidate

        db_path, conn = _setup_test_db()

        # Create mock papertrade_signals entry
        signal_id = "test_long_target"
        ticker = "TEST"
        entry = 100.0
        as_of = "2026-01-01T09:30:00"

        _insert_signal(conn, signal_id, ticker, "long", "1d", as_of, entry, 90.0, 175.0)

        # Create a synthetic price DataFrame with target hitting before stop
        # Entry at 100, target at 175, stop at 90
        # Day 2: open 101, low 99, high 180, close 160
        # (high 180 >= target 175, so target hits on bar 2)
        bar_data = {
            pd.Timestamp("2026-01-02"): {
                "Open": 101.0,
                "High": 180.0,
                "Low": 99.0,
                "Close": 160.0,
            },
            pd.Timestamp("2026-01-03"): {
                "Open": 161.0,
                "High": 165.0,
                "Low": 88.0,
                "Close": 155.0,
            }
        }
        bars_df = pd.DataFrame.from_dict(
            bar_data, orient="index"
        )
        bars_df.index.name = "timestamp"

        # Mock price_cache.get_price_data
        import scripts.tpsl_label_grid as tpsl_module
        original_get_price = tpsl_module.price_cache.get_price_data

        def mock_get_price(*args, **kwargs):
            return bars_df

        tpsl_module.price_cache.get_price_data = mock_get_price

        # Mock resolve_interval_for_signal
        original_resolve_interval = tpsl_module.resolve_interval_for_signal

        def mock_resolve_interval(*args, **kwargs):
            return "1d"

        tpsl_module.resolve_interval_for_signal = mock_resolve_interval

        try:
            # Resolve candidate with 10% stop, 75% target
            resolve_candidate(
                conn,
                {"signal_id": signal_id, "ticker": ticker, "direction": "long",
                 "entry": entry, "as_of": as_of},
                stop_pct=10.0,
                target_pct=75.0,
                interval_ladder=["1d"],
                fee_pct=0.001,
                slippage_pct=0.0005,
                db_path=db_path,
            )

            # Query result
            row = conn.execute(
                "SELECT * FROM tpsl_label_candidates WHERE signal_id=? AND stop_pct=? AND target_pct=?",
                (signal_id, 10.0, 75.0)
            ).fetchone()

            assert row is not None
            assert row["resolved"] == 1
            assert row["hit_target_first"] == 1  # Target hit first
            assert row["interval_used"] == "1d"

        finally:
            # Restore mocks
            tpsl_module.price_cache.get_price_data = original_get_price
            tpsl_module.resolve_interval_for_signal = original_resolve_interval
            conn.close()

    def test_long_stop_hit_first(self):
        """Long trade where stop hits before target."""
        from scripts.tpsl_label_grid import resolve_candidate

        db_path, conn = _setup_test_db()

        signal_id = "test_long_stop"
        ticker = "TEST"
        entry = 100.0
        as_of = "2026-01-01T09:30:00"

        _insert_signal(conn, signal_id, ticker, "long", "1d", as_of, entry, 90.0, 175.0)

        # Stop at 90, target at 175
        # Day 2: open 99, low 88, high 105, close 95
        # (low 88 <= stop 90, so stop hits on bar 2)
        bar_data = {
            pd.Timestamp("2026-01-02"): {
                "Open": 99.0,
                "High": 105.0,
                "Low": 88.0,
                "Close": 95.0,
            },
        }
        bars_df = pd.DataFrame.from_dict(bar_data, orient="index")
        bars_df.index.name = "timestamp"

        import scripts.tpsl_label_grid as tpsl_module
        original_get_price = tpsl_module.price_cache.get_price_data
        original_resolve_interval = tpsl_module.resolve_interval_for_signal

        def mock_get_price(*args, **kwargs):
            return bars_df

        def mock_resolve_interval(*args, **kwargs):
            return "1d"

        tpsl_module.price_cache.get_price_data = mock_get_price
        tpsl_module.resolve_interval_for_signal = mock_resolve_interval

        try:
            resolve_candidate(
                conn,
                {"signal_id": signal_id, "ticker": ticker, "direction": "long",
                 "entry": entry, "as_of": as_of},
                stop_pct=10.0,
                target_pct=75.0,
                interval_ladder=["1d"],
                fee_pct=0.001,
                slippage_pct=0.0005,
                db_path=db_path,
            )

            row = conn.execute(
                "SELECT * FROM tpsl_label_candidates WHERE signal_id=? AND stop_pct=? AND target_pct=?",
                (signal_id, 10.0, 75.0)
            ).fetchone()

            assert row is not None
            assert row["resolved"] == 1
            assert row["hit_target_first"] == 0  # Stop hit first
            assert row["interval_used"] == "1d"

        finally:
            tpsl_module.price_cache.get_price_data = original_get_price
            tpsl_module.resolve_interval_for_signal = original_resolve_interval
            conn.close()

    def test_candidate_never_resolves(self):
        """Candidate bracket never hits -- disqualified."""
        from scripts.tpsl_label_grid import resolve_candidate

        db_path, conn = _setup_test_db()

        signal_id = "test_no_resolve"
        ticker = "TEST"
        entry = 100.0
        as_of = "2026-01-01T09:30:00"

        _insert_signal(conn, signal_id, ticker, "long", "1d", as_of, entry, 90.0, 175.0)

        # Stop at 90, target at 175
        # Day 2: open 101, high 110, low 95, close 105
        # Day 3: open 105, high 108, low 100, close 102
        # Neither barrier is touched
        bar_data = {
            pd.Timestamp("2026-01-02"): {
                "Open": 101.0,
                "High": 110.0,
                "Low": 95.0,
                "Close": 105.0,
            },
            pd.Timestamp("2026-01-03"): {
                "Open": 105.0,
                "High": 108.0,
                "Low": 100.0,
                "Close": 102.0,
            }
        }
        bars_df = pd.DataFrame.from_dict(bar_data, orient="index")
        bars_df.index.name = "timestamp"

        import scripts.tpsl_label_grid as tpsl_module
        original_get_price = tpsl_module.price_cache.get_price_data
        original_resolve_interval = tpsl_module.resolve_interval_for_signal

        def mock_get_price(*args, **kwargs):
            return bars_df

        def mock_resolve_interval(*args, **kwargs):
            return "1d"

        tpsl_module.price_cache.get_price_data = mock_get_price
        tpsl_module.resolve_interval_for_signal = mock_resolve_interval

        try:
            resolve_candidate(
                conn,
                {"signal_id": signal_id, "ticker": ticker, "direction": "long",
                 "entry": entry, "as_of": as_of},
                stop_pct=10.0,
                target_pct=75.0,
                interval_ladder=["1d"],
                fee_pct=0.001,
                slippage_pct=0.0005,
                db_path=db_path,
            )

            row = conn.execute(
                "SELECT * FROM tpsl_label_candidates WHERE signal_id=? AND stop_pct=? AND target_pct=?",
                (signal_id, 10.0, 75.0)
            ).fetchone()

            assert row is not None
            assert row["resolved"] == 0  # Disqualified
            assert row["hit_target_first"] is None  # NULL as per spec
            assert row["interval_used"] is None

        finally:
            tpsl_module.price_cache.get_price_data = original_get_price
            tpsl_module.resolve_interval_for_signal = original_resolve_interval
            conn.close()


class TestIdempotency:
    """Test that re-running produces zero new/changed rows."""

    def test_idempotent_re_run(self):
        """Re-running with same engine_version produces zero new rows."""
        from scripts.tpsl_label_grid import resolve_candidate

        db_path, conn = _setup_test_db()

        signal_id = "test_idempotent"
        ticker = "TEST"
        entry = 100.0
        as_of = "2026-01-01T09:30:00"

        _insert_signal(conn, signal_id, ticker, "long", "1d", as_of, entry, 90.0, 175.0)

        # Setup mocks
        bar_data = {
            pd.Timestamp("2026-01-02"): {
                "Open": 101.0,
                "High": 180.0,
                "Low": 99.0,
                "Close": 160.0,
            },
        }
        bars_df = pd.DataFrame.from_dict(bar_data, orient="index")
        bars_df.index.name = "timestamp"

        import scripts.tpsl_label_grid as tpsl_module
        original_get_price = tpsl_module.price_cache.get_price_data
        original_resolve_interval = tpsl_module.resolve_interval_for_signal

        def mock_get_price(*args, **kwargs):
            return bars_df

        def mock_resolve_interval(*args, **kwargs):
            return "1d"

        tpsl_module.price_cache.get_price_data = mock_get_price
        tpsl_module.resolve_interval_for_signal = mock_resolve_interval

        try:
            # First run
            resolve_candidate(
                conn,
                {"signal_id": signal_id, "ticker": ticker, "direction": "long",
                 "entry": entry, "as_of": as_of},
                stop_pct=10.0,
                target_pct=75.0,
                interval_ladder=["1d"],
                fee_pct=0.001,
                slippage_pct=0.0005,
                db_path=db_path,
            )

            count_after_first = conn.execute(
                "SELECT COUNT(*) FROM tpsl_label_candidates"
            ).fetchone()[0]

            # Get the first row
            row1 = conn.execute(
                "SELECT * FROM tpsl_label_candidates WHERE signal_id=?",
                (signal_id,)
            ).fetchone()

            # Second run (idempotent)
            resolve_candidate(
                conn,
                {"signal_id": signal_id, "ticker": ticker, "direction": "long",
                 "entry": entry, "as_of": as_of},
                stop_pct=10.0,
                target_pct=75.0,
                interval_ladder=["1d"],
                fee_pct=0.001,
                slippage_pct=0.0005,
                db_path=db_path,
            )

            count_after_second = conn.execute(
                "SELECT COUNT(*) FROM tpsl_label_candidates"
            ).fetchone()[0]

            # Should be the same count (INSERT OR REPLACE, no new rows)
            assert count_after_first == count_after_second

            # Row should be identical
            row2 = conn.execute(
                "SELECT * FROM tpsl_label_candidates WHERE signal_id=?",
                (signal_id,)
            ).fetchone()

            assert row1["signal_id"] == row2["signal_id"]
            assert row1["hit_target_first"] == row2["hit_target_first"]
            assert row1["resolved"] == row2["resolved"]

        finally:
            tpsl_module.price_cache.get_price_data = original_get_price
            tpsl_module.resolve_interval_for_signal = original_resolve_interval
            conn.close()


class TestPriceCacheConfiguration:
    """Test that price_cache is configured before use."""

    def test_ensure_configured_db_called(self):
        """_ensure_configured_db should be called to configure local mode."""
        from scripts.tpsl_label_grid import resolve_candidate

        db_path, conn = _setup_test_db()

        signal_id = "test_config"
        ticker = "TEST"
        entry = 100.0
        as_of = "2026-01-01T09:30:00"

        _insert_signal(conn, signal_id, ticker, "long", "1d", as_of, entry, 90.0, 175.0)

        # Track calls to price_cache.configure
        import scripts.tpsl_label_grid as tpsl_module
        calls_to_configure = []

        original_configure = tpsl_module.price_cache.configure
        original_get_price = tpsl_module.price_cache.get_price_data

        def mock_configure(*args, **kwargs):
            calls_to_configure.append((args, kwargs))

        def mock_get_price(*args, **kwargs):
            bar_data = {
                pd.Timestamp("2026-01-02"): {
                    "Open": 101.0,
                    "High": 105.0,
                    "Low": 99.0,
                    "Close": 102.0,
                },
            }
            return pd.DataFrame.from_dict(bar_data, orient="index")

        tpsl_module.price_cache.configure = mock_configure
        tpsl_module.price_cache.get_price_data = mock_get_price

        try:
            # This should internally call _ensure_configured_db
            # which will call price_cache.configure
            resolve_candidate(
                conn,
                {"signal_id": signal_id, "ticker": ticker, "direction": "long",
                 "entry": entry, "as_of": as_of},
                stop_pct=10.0,
                target_pct=75.0,
                interval_ladder=["1d"],
                fee_pct=0.001,
                slippage_pct=0.0005,
                db_path=db_path,
            )

            # At least one configure call should have happened
            # via _ensure_configured_db in resolve_candidate
            assert len(calls_to_configure) > 0, \
                "price_cache.configure should have been called via _ensure_configured_db"

            # Verify remote=False was passed
            for args, kwargs in calls_to_configure:
                if "remote" in kwargs:
                    assert kwargs["remote"] is False, \
                        "price_cache.configure should be called with remote=False"

        finally:
            tpsl_module.price_cache.configure = original_configure
            tpsl_module.price_cache.get_price_data = original_get_price
            conn.close()


class TestResumeLogic:
    """Test that the resume query correctly handles partial coverage."""

    def test_partial_coverage_resumed(self):
        """Resume should re-process a signal with only 2 of 9 candidates written.

        Simulates an interrupted run: signal has 2 of 9 candidate rows already
        persisted for the current engine_version. The resume query should
        include this signal in uncovered_signals, so remaining 7 candidates
        get computed.
        """
        db_path, conn = _setup_test_db()

        # Create a signal
        signal_id = "partial_coverage"
        ticker = "TEST"
        entry = 100.0
        as_of = "2026-01-01T09:30:00"
        _insert_signal(conn, signal_id, ticker, "long", "1d", as_of, entry, 90.0, 175.0)

        # Pre-populate 2 of 9 candidates (simulating interrupted run)
        import scripts.tpsl_label_grid as tpsl_module
        engine_version = tpsl_module.ENGINE_VERSION
        computed_at = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            INSERT INTO tpsl_label_candidates (
                signal_id, stop_pct, target_pct, resolved, hit_target_first,
                interval_used, engine_version, computed_at
            ) VALUES (?, ?, ?, 1, 1, '1d', ?, ?)
            """,
            (signal_id, 10.0, 75.0, engine_version, computed_at)
        )
        conn.execute(
            """
            INSERT INTO tpsl_label_candidates (
                signal_id, stop_pct, target_pct, resolved, hit_target_first,
                interval_used, engine_version, computed_at
            ) VALUES (?, ?, ?, 1, 0, '1d', ?, ?)
            """,
            (signal_id, 15.0, 85.0, engine_version, computed_at)
        )
        conn.commit()

        # Verify only 2 rows exist for this signal
        count_before = conn.execute(
            "SELECT COUNT(*) FROM tpsl_label_candidates WHERE signal_id=?",
            (signal_id,)
        ).fetchone()[0]
        assert count_before == 2, f"Expected 2 initial rows, got {count_before}"

        # Test the resume query
        expected_grid_size = len(tpsl_module.STOP_PCT_GRID) * len(tpsl_module.TARGET_PCT_GRID)
        cursor = conn.execute(
            """
            SELECT DISTINCT s.signal_id, s.ticker, s.direction, s.entry, s.as_of
            FROM papertrade_signals s
            LEFT JOIN (
                SELECT signal_id
                FROM tpsl_label_candidates
                WHERE engine_version = ?
                GROUP BY signal_id
                HAVING COUNT(*) = ?
            ) c ON s.signal_id = c.signal_id
            WHERE c.signal_id IS NULL
            ORDER BY s.as_of
            """,
            (engine_version, expected_grid_size)
        )
        uncovered = cursor.fetchall()

        # Signal with only 2 of 9 rows should be in uncovered list
        signal_ids = [row[0] for row in uncovered]
        assert signal_id in signal_ids, \
            f"Signal {signal_id} should be in uncovered list (only 2 of {expected_grid_size} rows)"

        conn.close()

    def test_full_coverage_skipped(self):
        """Resume should skip a signal with all 9 candidates already written.

        Verifies that a fully-covered signal is NOT included in uncovered_signals,
        confirming no wasted recomputation.
        """
        db_path, conn = _setup_test_db()

        # Create a signal
        signal_id = "full_coverage"
        ticker = "TEST"
        entry = 100.0
        as_of = "2026-01-01T09:30:00"
        _insert_signal(conn, signal_id, ticker, "long", "1d", as_of, entry, 90.0, 175.0)

        # Pre-populate ALL 9 candidates
        import scripts.tpsl_label_grid as tpsl_module
        engine_version = tpsl_module.ENGINE_VERSION
        computed_at = datetime.now(timezone.utc).isoformat()

        for stop_pct in tpsl_module.STOP_PCT_GRID:
            for target_pct in tpsl_module.TARGET_PCT_GRID:
                conn.execute(
                    """
                    INSERT INTO tpsl_label_candidates (
                        signal_id, stop_pct, target_pct, resolved, hit_target_first,
                        interval_used, engine_version, computed_at
                    ) VALUES (?, ?, ?, 1, 1, '1d', ?, ?)
                    """,
                    (signal_id, stop_pct, target_pct, engine_version, computed_at)
                )
        conn.commit()

        # Verify all 9 rows exist for this signal
        count_before = conn.execute(
            "SELECT COUNT(*) FROM tpsl_label_candidates WHERE signal_id=?",
            (signal_id,)
        ).fetchone()[0]
        expected_grid_size = len(tpsl_module.STOP_PCT_GRID) * len(tpsl_module.TARGET_PCT_GRID)
        assert count_before == expected_grid_size, \
            f"Expected {expected_grid_size} rows, got {count_before}"

        # Test the resume query
        cursor = conn.execute(
            """
            SELECT DISTINCT s.signal_id, s.ticker, s.direction, s.entry, s.as_of
            FROM papertrade_signals s
            LEFT JOIN (
                SELECT signal_id
                FROM tpsl_label_candidates
                WHERE engine_version = ?
                GROUP BY signal_id
                HAVING COUNT(*) = ?
            ) c ON s.signal_id = c.signal_id
            WHERE c.signal_id IS NULL
            ORDER BY s.as_of
            """,
            (engine_version, expected_grid_size)
        )
        uncovered = cursor.fetchall()

        # Signal with full coverage should NOT be in uncovered list
        signal_ids = [row[0] for row in uncovered]
        assert signal_id not in signal_ids, \
            f"Fully-covered signal {signal_id} should NOT be in uncovered list"

        conn.close()
