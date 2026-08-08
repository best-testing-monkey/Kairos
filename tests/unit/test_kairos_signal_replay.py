"""Unit tests for kairos_signal_replay module."""

import sqlite3
import sys
import os
import json
import hashlib
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from kairos_signal_replay import (
    _ensure_signal_replay_tables,
    unpack_signals_cache_to_papertrade_signals,
    resolve_interval_for_signal
)


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


def _build_signals_cache_table(conn) -> None:
    """Helper to create signals_cache table in test DB."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signals_cache (
            cache_key TEXT PRIMARY KEY,
            strategy_name TEXT NOT NULL,
            assets TEXT NOT NULL,
            interval TEXT NOT NULL,
            as_of TEXT NOT NULL,
            lookback INTEGER NOT NULL,
            pred_samples INTEGER NOT NULL,
            min_ev_pct REAL NOT NULL,
            model_label TEXT NOT NULL,
            model_path TEXT,
            checkpoint_fingerprint TEXT NOT NULL DEFAULT '',
            stats_json TEXT NOT NULL,
            advice_json TEXT NOT NULL,
            skipped_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def test_unpack_signals_cache_basic():
    """Verify unpack_signals_cache_to_papertrade_signals unpacks signals correctly.

    Includes one FLAT direction (to verify exclusion) and one normal LONG signal
    (to verify inclusion with correct field values).
    """
    conn = sqlite3.connect(":memory:")
    _build_signals_cache_table(conn)
    _ensure_signal_replay_tables(conn)

    # Build synthetic signals_cache row with mixed FLAT and LONG signals
    stats_json = json.dumps([
        {
            "strategy": "test_strategy_1",
            "symbol": "BTC-USD",
            "direction": "FLAT",
            "entry": 100.0,
            "stop": 99.0,
            "target": 101.0,
            "expected_value": 0.5,
            "base_win_rate": 0.6,
        },
        {
            "strategy": "test_strategy_1",
            "symbol": "ETH-USD",
            "direction": "LONG",
            "entry": 50.0,
            "stop": 49.0,
            "target": 51.0,
            "expected_value": 0.25,
            "base_win_rate": 0.65,
        },
    ])

    advice_json = json.dumps([
        {
            "expected_value": 0.5,
            "base_signals": 10,
            "oracle_signals": 8,
        },
        {
            "expected_value": 0.25,
            "base_signals": 20,
            "oracle_signals": 15,
        },
    ])

    cache_key = "test_cache_key_1"
    strategy_name = "test_strategy_1"
    assets = "BTC-USD,ETH-USD"
    interval = "1d"
    as_of = "2026-08-07"
    model_label = "base"
    checkpoint_fingerprint = ""

    conn.execute(
        """
        INSERT INTO signals_cache (
            cache_key, strategy_name, assets, interval, as_of, lookback,
            pred_samples, min_ev_pct, model_label, model_path,
            checkpoint_fingerprint, stats_json, advice_json, skipped_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (cache_key, strategy_name, assets, interval, as_of, 100, 50, 0.0,
         model_label, None, checkpoint_fingerprint, stats_json, advice_json,
         "[]", "2026-08-07T10:00:00Z")
    )
    conn.commit()

    # Call unpack function
    inserted = unpack_signals_cache_to_papertrade_signals(
        conn, "2026-08-01", "2026-08-31"
    )

    # Should insert 1 row (FLAT is excluded, LONG is included)
    assert inserted == 1, f"Expected 1 row inserted, got {inserted}"

    # Verify papertrade_signals table has the expected row
    cursor = conn.execute("SELECT * FROM papertrade_signals")
    rows = cursor.fetchall()
    assert len(rows) == 1, f"Expected 1 row in papertrade_signals, got {len(rows)}"

    # Get column names
    cursor_desc = conn.execute("SELECT * FROM papertrade_signals LIMIT 0")
    column_names = [desc[0] for desc in cursor_desc.description]

    # Convert row to dict
    row_dict = dict(zip(column_names, rows[0]))

    # Verify LONG signal was inserted with correct values
    assert row_dict["strategy_name"] == "test_strategy_1"
    assert row_dict["ticker"] == "ETH-USD"
    assert row_dict["direction"] == "long"
    assert row_dict["interval"] == "1d"
    assert row_dict["as_of"] == "2026-08-07"
    assert row_dict["entry"] == 50.0
    assert row_dict["stop"] == 49.0
    assert row_dict["target"] == 51.0
    assert row_dict["expected_value"] == 0.25
    assert row_dict["base_win_rate"] == 0.65
    assert row_dict["n"] == 20  # base_signals value
    assert row_dict["model_label"] == "base"
    assert row_dict["checkpoint_fingerprint"] == ""
    assert row_dict["source_cache_key"] == cache_key
    assert row_dict["signal_id"] is not None
    assert len(row_dict["signal_id"]) == 64  # SHA256 hex is 64 chars

    conn.close()


def test_unpack_signals_cache_idempotent():
    """Verify that calling unpack twice over same window returns 0 on second call."""
    conn = sqlite3.connect(":memory:")
    _build_signals_cache_table(conn)
    _ensure_signal_replay_tables(conn)

    # Insert one signals_cache row
    stats_json = json.dumps([
        {
            "strategy": "test_strategy_2",
            "symbol": "SOL-USD",
            "direction": "SHORT",
            "entry": 25.0,
            "stop": 26.0,
            "target": 24.0,
            "expected_value": -0.1,
            "base_win_rate": 0.55,
        },
    ])

    advice_json = json.dumps([
        {
            "expected_value": -0.1,
            "base_signals": None,
            "oracle_signals": 12,
        },
    ])

    conn.execute(
        """
        INSERT INTO signals_cache (
            cache_key, strategy_name, assets, interval, as_of, lookback,
            pred_samples, min_ev_pct, model_label, model_path,
            checkpoint_fingerprint, stats_json, advice_json, skipped_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("test_cache_key_2", "test_strategy_2", "SOL-USD", "1d",
         "2026-08-07", 100, 50, 0.0, "base", None, "", stats_json,
         advice_json, "[]", "2026-08-07T11:00:00Z")
    )
    conn.commit()

    # First call
    inserted_first = unpack_signals_cache_to_papertrade_signals(
        conn, "2026-08-01", "2026-08-31"
    )
    assert inserted_first == 1, f"First call should insert 1 row, got {inserted_first}"

    # Verify one row exists
    cursor = conn.execute("SELECT COUNT(*) FROM papertrade_signals")
    count_after_first = cursor.fetchone()[0]
    assert count_after_first == 1

    # Second call (idempotent)
    inserted_second = unpack_signals_cache_to_papertrade_signals(
        conn, "2026-08-01", "2026-08-31"
    )
    assert inserted_second == 0, \
        f"Second call over same window should insert 0 rows, got {inserted_second}"

    # Verify count unchanged
    cursor = conn.execute("SELECT COUNT(*) FROM papertrade_signals")
    count_after_second = cursor.fetchone()[0]
    assert count_after_second == 1, "Row count should remain 1"

    conn.close()


def test_unpack_signals_cache_oracle_signals_fallback():
    """Verify that n uses oracle_signals when base_signals is None."""
    conn = sqlite3.connect(":memory:")
    _build_signals_cache_table(conn)
    _ensure_signal_replay_tables(conn)

    # Insert signals_cache row where base_signals is None, oracle_signals is set
    stats_json = json.dumps([
        {
            "strategy": "test_strategy_3",
            "symbol": "ADA-USD",
            "direction": "LONG",
            "entry": 1.0,
            "stop": 0.95,
            "target": 1.05,
        },
    ])

    advice_json = json.dumps([
        {
            "base_signals": None,
            "oracle_signals": 7,
        },
    ])

    conn.execute(
        """
        INSERT INTO signals_cache (
            cache_key, strategy_name, assets, interval, as_of, lookback,
            pred_samples, min_ev_pct, model_label, model_path,
            checkpoint_fingerprint, stats_json, advice_json, skipped_json,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("test_cache_key_3", "test_strategy_3", "ADA-USD", "1d",
         "2026-08-06", 100, 50, 0.0, "base", None, "", stats_json,
         advice_json, "[]", "2026-08-06T12:00:00Z")
    )
    conn.commit()

    inserted = unpack_signals_cache_to_papertrade_signals(
        conn, "2026-08-01", "2026-08-31"
    )
    assert inserted == 1

    cursor = conn.execute("SELECT n FROM papertrade_signals")
    n_value = cursor.fetchone()[0]
    assert n_value == 7, f"Expected n=7 (oracle_signals), got {n_value}"

    conn.close()


def test_unpack_signals_cache_deterministic_signal_id():
    """Verify that signal_id is deterministic and collision-resistant."""
    conn = sqlite3.connect(":memory:")
    _build_signals_cache_table(conn)
    _ensure_signal_replay_tables(conn)

    stats_json = json.dumps([
        {
            "strategy": "test_strategy_4",
            "symbol": "XRP-USD",
            "direction": "LONG",
            "entry": 3.0,
            "stop": 2.95,
            "target": 3.05,
        },
    ])

    advice_json = json.dumps([
        {"base_signals": 5, "oracle_signals": 4},
    ])

    # Insert twice
    for i in range(2):
        conn.execute(
            """
            INSERT INTO signals_cache (
                cache_key, strategy_name, assets, interval, as_of, lookback,
                pred_samples, min_ev_pct, model_label, model_path,
                checkpoint_fingerprint, stats_json, advice_json, skipped_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"test_cache_key_4_{i}", "test_strategy_4", "XRP-USD", "1d",
             "2026-08-05", 100, 50, 0.0, "base", None, "", stats_json,
             advice_json, "[]", "2026-08-05T13:00:00Z")
        )
    conn.commit()

    # Unpack both rows
    inserted = unpack_signals_cache_to_papertrade_signals(
        conn, "2026-08-01", "2026-08-31"
    )

    # Both should map to the same signal_id, so only 1 inserted (second is IGNORE)
    assert inserted == 1, \
        f"Both rows should produce same signal_id, expected 1 inserted, got {inserted}"

    # Verify only 1 row in papertrade_signals
    cursor = conn.execute("SELECT COUNT(DISTINCT signal_id) FROM papertrade_signals")
    unique_count = cursor.fetchone()[0]
    assert unique_count == 1, f"Expected 1 unique signal_id, got {unique_count}"

    conn.close()


# ==============================================================================
# Tests for resolve_interval_for_signal
# ==============================================================================


def _make_price_df(num_bars: int) -> pd.DataFrame:
    """Helper to create a synthetic OHLCV DataFrame with num_bars rows."""
    dates = pd.date_range(start="2026-08-01", periods=num_bars, freq="D", tz="UTC")
    return pd.DataFrame({
        "Open": [100.0] * num_bars,
        "High": [101.0] * num_bars,
        "Low": [99.0] * num_bars,
        "Close": [100.5] * num_bars,
        "Volume": [1000000] * num_bars,
    }, index=dates)


def test_resolve_interval_sufficient_data_first_interval():
    """Test that the function returns the first interval when it has sufficient data.

    Verifies that price_cache.get_price_data is called exactly once (no unnecessary
    fallback attempts).
    """
    entry_dt = datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc)

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        # First interval has 10 bars (more than min_bars=2)
        mock_get.return_value = _make_price_df(10)

        result = resolve_interval_for_signal(
            ticker="AAPL",
            entry_datetime=entry_dt,
            interval_ladder=["1h", "4h", "1d"],
            min_bars=2,
            db_path="/tmp/test.db"
        )

        assert result == "1h", f"Expected '1h', got {result}"
        # Should only call price_cache once (for the first interval)
        assert mock_get.call_count == 1, (
            f"Expected 1 call to price_cache, got {mock_get.call_count}"
        )


def test_resolve_interval_fallback_to_second():
    """Test fallback when the smallest interval has insufficient data.

    The first interval returns an empty DataFrame, the second returns sufficient
    data. Verifies that the second interval is returned and price_cache is called
    twice (once for each interval tried).
    """
    entry_dt = datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc)

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        # First interval: empty
        # Second interval: 5 bars
        mock_get.side_effect = [
            pd.DataFrame(),  # 1h returns empty
            _make_price_df(5)  # 4h returns 5 bars
        ]

        result = resolve_interval_for_signal(
            ticker="BTC-USD",
            entry_datetime=entry_dt,
            interval_ladder=["1h", "4h", "1d"],
            min_bars=2,
            db_path="/tmp/test.db"
        )

        assert result == "4h", f"Expected '4h', got {result}"
        assert mock_get.call_count == 2, (
            f"Expected 2 calls to price_cache, got {mock_get.call_count}"
        )


def test_resolve_interval_all_empty():
    """Test that None is returned when all intervals yield insufficient data."""
    entry_dt = datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc)

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        # All intervals return either None or empty
        mock_get.side_effect = [
            None,  # 1h returns None
            pd.DataFrame(),  # 4h returns empty
            _make_price_df(1)  # 1d returns only 1 bar (less than min_bars=2)
        ]

        result = resolve_interval_for_signal(
            ticker="ETH-USD",
            entry_datetime=entry_dt,
            interval_ladder=["1h", "4h", "1d"],
            min_bars=2,
            db_path="/tmp/test.db"
        )

        assert result is None, f"Expected None, got {result}"
        assert mock_get.call_count == 3, (
            f"Expected 3 calls (one per interval), got {mock_get.call_count}"
        )


def test_resolve_interval_exception_handling():
    """Test that exceptions in price_cache calls are caught and don't propagate.

    When one interval's call raises an exception, the function moves to the next
    interval without propagating the exception.
    """
    entry_dt = datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc)

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        # First interval raises exception
        # Second interval succeeds
        mock_get.side_effect = [
            ValueError("Network error"),  # 1h raises
            _make_price_df(8)  # 4h succeeds with 8 bars
        ]

        result = resolve_interval_for_signal(
            ticker="SOL-USD",
            entry_datetime=entry_dt,
            interval_ladder=["1h", "4h", "1d"],
            min_bars=2,
            db_path="/tmp/test.db"
        )

        assert result == "4h", f"Expected '4h', got {result}"
        assert mock_get.call_count == 2, (
            f"Expected 2 calls to price_cache, got {mock_get.call_count}"
        )


def test_resolve_interval_empty_ladder():
    """Test that None is returned when interval_ladder is empty."""
    entry_dt = datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc)

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        result = resolve_interval_for_signal(
            ticker="AAPL",
            entry_datetime=entry_dt,
            interval_ladder=[],
            min_bars=2,
            db_path="/tmp/test.db"
        )

        assert result is None, f"Expected None for empty ladder, got {result}"
        assert mock_get.call_count == 0, (
            f"Expected no calls for empty ladder, got {mock_get.call_count}"
        )


def test_resolve_interval_empty_ticker():
    """Test that None is returned when ticker is empty."""
    entry_dt = datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc)

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        result = resolve_interval_for_signal(
            ticker="",
            entry_datetime=entry_dt,
            interval_ladder=["1h", "4h"],
            min_bars=2,
            db_path="/tmp/test.db"
        )

        assert result is None, f"Expected None for empty ticker, got {result}"
        assert mock_get.call_count == 0, (
            f"Expected no calls for empty ticker, got {mock_get.call_count}"
        )


def test_resolve_interval_min_bars_boundary():
    """Test that min_bars boundary is respected (exactly min_bars should pass)."""
    entry_dt = datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc)

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        # First interval returns exactly min_bars (should succeed)
        mock_get.return_value = _make_price_df(3)

        result = resolve_interval_for_signal(
            ticker="XRP-USD",
            entry_datetime=entry_dt,
            interval_ladder=["1h", "4h"],
            min_bars=3,
            db_path="/tmp/test.db"
        )

        assert result == "1h", f"Expected '1h' with exactly min_bars, got {result}"
        assert mock_get.call_count == 1, (
            f"Expected 1 call, got {mock_get.call_count}"
        )


def test_resolve_interval_min_bars_default():
    """Test that min_bars defaults to 2."""
    entry_dt = datetime(2026, 8, 7, 10, 0, 0, tzinfo=timezone.utc)

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        # Return exactly 2 bars
        mock_get.return_value = _make_price_df(2)

        result = resolve_interval_for_signal(
            ticker="ADA-USD",
            entry_datetime=entry_dt,
            interval_ladder=["1h"],
            db_path="/tmp/test.db"
            # No min_bars specified; should default to 2
        )

        assert result == "1h", f"Expected '1h' with default min_bars=2, got {result}"
