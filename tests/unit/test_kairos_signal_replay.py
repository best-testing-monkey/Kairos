"""Unit tests for kairos_signal_replay module."""

import sqlite3
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from kairos_signal_replay import _ensure_signal_replay_tables


def test_ensure_signal_replay_tables_creates_tables():
    """Verify that _ensure_signal_replay_tables creates both tables with
    the correct schema."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # Check papertrade_signals table exists and has expected columns
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='papertrade_signals'"
    )
    assert cursor.fetchone() is not None, "papertrade_signals table not created"

    # Check papertrade_signals_closure table exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='papertrade_signals_closure'"
    )
    assert cursor.fetchone() is not None, "papertrade_signals_closure table not created"

    # Check idx_papertrade_signals_as_of index exists
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_papertrade_signals_as_of'"
    )
    assert cursor.fetchone() is not None, "idx_papertrade_signals_as_of index not created"

    conn.close()


def test_ensure_signal_replay_tables_column_schema():
    """Verify papertrade_signals has all expected columns."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # Get column info for papertrade_signals
    cursor = conn.execute("PRAGMA table_info(papertrade_signals)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}  # name: type

    expected_columns = {
        "signal_id", "strategy_name", "ticker", "direction", "interval",
        "as_of", "entry", "stop", "target", "expected_value", "base_win_rate",
        "n", "model_label", "checkpoint_fingerprint", "source_cache_key",
        "created_at"
    }
    assert set(columns.keys()) == expected_columns, \
        f"papertrade_signals columns mismatch: got {set(columns.keys())}"

    conn.close()


def test_ensure_signal_replay_tables_closure_column_schema():
    """Verify papertrade_signals_closure has all expected columns."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # Get column info for papertrade_signals_closure
    cursor = conn.execute("PRAGMA table_info(papertrade_signals_closure)")
    columns = {row[1]: row[2] for row in cursor.fetchall()}  # name: type

    expected_columns = {
        "signal_id", "resolved", "interval_used", "pct_profit",
        "max_drawdown_pct", "trigger_datetime", "exit_datetime",
        "exit_reason", "computed_at", "engine_version"
    }
    assert set(columns.keys()) == expected_columns, \
        f"papertrade_signals_closure columns mismatch: got {set(columns.keys())}"

    conn.close()


def test_ensure_signal_replay_tables_idempotent():
    """Verify that calling _ensure_signal_replay_tables multiple times
    does not raise an error."""
    conn = sqlite3.connect(":memory:")

    # First call
    _ensure_signal_replay_tables(conn)

    # Second call should not raise
    _ensure_signal_replay_tables(conn)

    # Verify tables still exist
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
        "('papertrade_signals', 'papertrade_signals_closure')"
    )
    tables = {row[0] for row in cursor.fetchall()}
    assert tables == {"papertrade_signals", "papertrade_signals_closure"}

    conn.close()
