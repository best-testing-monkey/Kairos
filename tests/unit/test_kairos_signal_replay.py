"""Unit tests for kairos_signal_replay module."""

import sqlite3
import sys
import os
import json
import hashlib
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import price_cache

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from kairos_signal_replay import (
    _ensure_signal_replay_tables,
    unpack_signals_cache_to_papertrade_signals,
    resolve_interval_for_signal,
    max_adverse_excursion_pct,
    compute_closure,
    compute_closures_for_window,
    replay_steps,
    load_step_candidates,
    replay,
    _build_arg_parser,
)
from allocation import AllocationConfig


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


# ==============================================================================
# Tests for max_adverse_excursion_pct
# ==============================================================================


def test_max_adverse_excursion_pct_long_direction_hand_computed():
    """Test long position with hand-computed worst-case value.

    Setup: entry_price=100, bars with Low=[98, 95, 97]
    - Bar 1: (98-100)/100*100 = -2.0%
    - Bar 2: (95-100)/100*100 = -5.0%  <-- worst
    - Bar 3: (97-100)/100*100 = -3.0%
    Expected return: 5.0 (magnitude of worst move)
    """
    entry_price = 100.0
    bars = pd.DataFrame({
        "Open": [99.0, 94.0, 96.0],
        "High": [101.0, 99.0, 100.0],
        "Low": [98.0, 95.0, 97.0],
        "Close": [99.5, 96.0, 98.0],
        "Volume": [1000000, 1000000, 1000000],
    })

    result = max_adverse_excursion_pct("long", entry_price, bars)
    assert result == 5.0, (
        f"Long position: expected 5.0 (worst Low=95), got {result}"
    )


def test_max_adverse_excursion_pct_short_direction_hand_computed():
    """Test short position with hand-computed worst-case value.

    Setup: entry_price=100, bars with High=[101, 105, 102]
    - Bar 1: (100-101)/100*100 = -1.0%
    - Bar 2: (100-105)/100*100 = -5.0%  <-- worst
    - Bar 3: (100-102)/100*100 = -2.0%
    Expected return: 5.0 (magnitude of worst move)
    """
    entry_price = 100.0
    bars = pd.DataFrame({
        "Open": [101.0, 104.0, 101.0],
        "High": [101.0, 105.0, 102.0],
        "Low": [99.0, 103.0, 100.0],
        "Close": [100.5, 104.0, 101.5],
        "Volume": [1000000, 1000000, 1000000],
    })

    result = max_adverse_excursion_pct("short", entry_price, bars)
    assert result == 5.0, (
        f"Short position: expected 5.0 (worst High=105), got {result}"
    )


def test_max_adverse_excursion_pct_long_never_underwater():
    """Test long position that never goes underwater.

    All bars have Low >= entry_price; expected return is 0.0.
    """
    entry_price = 100.0
    bars = pd.DataFrame({
        "Open": [100.5, 102.0, 101.0],
        "High": [102.0, 104.0, 103.0],
        "Low": [100.0, 101.5, 100.5],  # All >= entry_price
        "Close": [101.0, 103.0, 102.0],
        "Volume": [1000000, 1000000, 1000000],
    })

    result = max_adverse_excursion_pct("long", entry_price, bars)
    assert result == 0.0, (
        f"Long position never underwater: expected 0.0, got {result}"
    )


def test_max_adverse_excursion_pct_short_never_underwater():
    """Test short position that never goes underwater.

    All bars have High <= entry_price; expected return is 0.0.
    """
    entry_price = 100.0
    bars = pd.DataFrame({
        "Open": [99.5, 98.0, 99.0],
        "High": [100.0, 98.5, 99.5],  # All <= entry_price
        "Low": [98.0, 96.5, 98.5],
        "Close": [99.0, 97.0, 99.0],
        "Volume": [1000000, 1000000, 1000000],
    })

    result = max_adverse_excursion_pct("short", entry_price, bars)
    assert result == 0.0, (
        f"Short position never underwater: expected 0.0, got {result}"
    )


def test_max_adverse_excursion_pct_empty_bars():
    """Test empty bars DataFrame returns 0.0 without raising."""
    entry_price = 100.0
    bars = pd.DataFrame({
        "Open": [],
        "High": [],
        "Low": [],
        "Close": [],
        "Volume": [],
    })

    result = max_adverse_excursion_pct("long", entry_price, bars)
    assert result == 0.0, (
        f"Empty bars: expected 0.0, got {result}"
    )

    result = max_adverse_excursion_pct("short", entry_price, bars)
    assert result == 0.0, (
        f"Empty bars (short): expected 0.0, got {result}"
    )


def test_max_adverse_excursion_pct_invalid_direction():
    """Test that invalid direction raises ValueError."""
    entry_price = 100.0
    bars = pd.DataFrame({
        "Open": [100.0],
        "High": [101.0],
        "Low": [99.0],
        "Close": [100.5],
        "Volume": [1000000],
    })

    with pytest.raises(ValueError, match="direction must be"):
        max_adverse_excursion_pct("up", entry_price, bars)

    with pytest.raises(ValueError, match="direction must be"):
        max_adverse_excursion_pct("LONG", entry_price, bars)


# ==============================================================================
# Tests for compute_closure / compute_closures_for_window
# ==============================================================================


def _insert_papertrade_signal(
    conn,
    signal_id: str,
    ticker: str = "AAPL",
    direction: str = "long",
    interval: str = "1h",
    as_of: str = "2026-08-01",
    entry: float = 100.0,
    stop: float = 95.0,
    target: float = 105.0,
) -> None:
    """Helper: insert a minimal papertrade_signals row for closure tests."""
    conn.execute(
        """
        INSERT INTO papertrade_signals (
            signal_id, strategy_name, ticker, direction, interval, as_of,
            entry, stop, target, expected_value, base_win_rate, n,
            model_label, checkpoint_fingerprint, source_cache_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (signal_id, "test_strategy", ticker, direction, interval, as_of,
         entry, stop, target, None, None, None, "base", "", None,
         "2026-08-01T00:00:00Z")
    )
    conn.commit()


def test_compute_closure_resolves_at_target_hand_computed():
    """Resolvable long signal that hits target on the second forward bar.

    Setup: entry=100, stop=95, target=110, direction=long, fee_pct=0.001.

    Hand derivation of _check_exit's bar-by-bar walk (LONG branch order:
    open-gap-through-stop, open-gap-through-target, intrabar stop touch,
    intrabar target touch, else fall back to close):
    - Bar 1 (Open=100.5, High=104, Low=99, Close=102): open not <=95 or
      >=110; low=99 not <=95; high=104 not >=110 -> falls back to
      (close=102, "close"). "close" is treated as NON-terminal here (see
      _TERMINAL_EXIT_REASONS), so the walk continues to bar 2.
      MAE at bar 1: (Low - entry) / entry * 100 = (99-100)/100*100 = -1.0%.
    - Bar 2 (Open=103, High=111, Low=101, Close=109): open not <=95 or
      >=110; low=101 not <=95; high=111 >= 110 -> exit at (target=110,
      "target"). Terminal, loop stops here.
      MAE at bar 2: (101-100)/100*100 = +1.0% (not worse than -1.0%).
    - worst_pct_move = -1.0% -> max_drawdown_pct = 1.0.
    - pnl = (exit_price - entry) * size - exit_price * size * fee_pct
          = (110 - 100) * 1.0 - 110 * 1.0 * 0.001 = 10 - 0.11 = 9.89
    - pct_profit = pnl / (entry * size) * 100.0 = 9.89 / 100 * 100.0 = 9.89
      (same "percentage number" scale as max_drawdown_pct, e.g. 9.89 means
      a 9.89% gain, not 0.0989)
    """
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    dates = pd.date_range(start="2026-08-01T00:00:00", periods=2, freq="h", tz="UTC")
    bars_df = pd.DataFrame({
        "Open": [100.5, 103.0],
        "High": [104.0, 111.0],
        "Low": [99.0, 101.0],
        "Close": [102.0, 109.0],
        "Volume": [1000, 1000],
    }, index=dates)

    signal_row = {
        "signal_id": "sig_target_hit",
        "ticker": "AAPL",
        "direction": "long",
        "entry": 100.0,
        "stop": 95.0,
        "target": 110.0,
        "as_of": "2026-08-01",
    }

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = bars_df
        compute_closure(
            conn, signal_row, interval_ladder=["1h"],
            fee_pct=0.001, slippage_pct=0.0005,
            db_path="/tmp/test.db", engine_version="v1",
        )

    cursor = conn.execute(
        "SELECT resolved, interval_used, pct_profit, max_drawdown_pct, "
        "exit_reason, engine_version, trigger_datetime, exit_datetime "
        "FROM papertrade_signals_closure WHERE signal_id = ?",
        ("sig_target_hit",)
    )
    row = cursor.fetchone()
    assert row is not None, "Expected a closure row to be written"
    (resolved, interval_used, pct_profit, max_drawdown_pct,
     exit_reason, engine_version, trigger_datetime, exit_datetime) = row

    assert resolved == 1
    assert interval_used == "1h"
    assert pct_profit == pytest.approx(9.89)
    assert max_drawdown_pct == pytest.approx(1.0)
    assert exit_reason == "target"
    assert engine_version == "v1"
    assert trigger_datetime is not None
    assert exit_datetime is not None

    conn.close()


def test_compute_closure_disqualified_no_interval_resolves():
    """Disqualification path 1: every interval in the ladder fails to resolve
    (price_cache.get_price_data returns empty for all of them). No exit walk
    is ever attempted; the closure row is resolved=0 with every other column
    NULL except engine_version/computed_at."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    signal_row = {
        "signal_id": "sig_no_interval",
        "ticker": "ZZZ",
        "direction": "long",
        "entry": 10.0,
        "stop": 9.0,
        "target": 11.0,
        "as_of": "2026-08-01",
    }

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = pd.DataFrame()  # empty for every interval tried
        compute_closure(
            conn, signal_row, interval_ladder=["1h", "4h", "1d"],
            fee_pct=0.001, slippage_pct=0.0005,
            db_path="/tmp/test.db", engine_version="v1",
        )

    cursor = conn.execute(
        "SELECT resolved, interval_used, pct_profit, max_drawdown_pct, "
        "trigger_datetime, exit_datetime, exit_reason, engine_version "
        "FROM papertrade_signals_closure WHERE signal_id = ?",
        ("sig_no_interval",)
    )
    row = cursor.fetchone()
    assert row is not None
    (resolved, interval_used, pct_profit, max_drawdown_pct,
     trigger_datetime, exit_datetime, exit_reason, engine_version) = row

    assert resolved == 0
    assert interval_used is None
    assert pct_profit is None
    assert max_drawdown_pct is None
    assert trigger_datetime is None
    assert exit_datetime is None
    assert exit_reason is None
    assert engine_version == "v1"

    conn.close()


def test_compute_closure_disqualified_bars_exhausted_without_resolving():
    """Disqualification path 2: the interval resolves and bars are fetched,
    but neither ever triggers a genuine stop/target/gap exit -- every bar
    falls back to _check_exit's ("close", "close") result, which this module
    treats as non-terminal. The walk exhausts all fetched bars and the
    signal is disqualified the same way as an interval-resolution failure.
    """
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # entry=100, stop=90, target=120 -- bars stay strictly inside that band,
    # never gapping through or touching stop/target intrabar.
    dates = pd.date_range(start="2026-08-01T00:00:00", periods=2, freq="h", tz="UTC")
    bars_df = pd.DataFrame({
        "Open": [101.0, 104.0],
        "High": [105.0, 107.0],
        "Low": [98.0, 99.0],
        "Close": [103.0, 105.0],
        "Volume": [1000, 1000],
    }, index=dates)

    signal_row = {
        "signal_id": "sig_exhausted",
        "ticker": "AAPL",
        "direction": "long",
        "entry": 100.0,
        "stop": 90.0,
        "target": 120.0,
        "as_of": "2026-08-01",
    }

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = bars_df
        compute_closure(
            conn, signal_row, interval_ladder=["1h"],
            fee_pct=0.001, slippage_pct=0.0005,
            db_path="/tmp/test.db", engine_version="v1",
        )

    cursor = conn.execute(
        "SELECT resolved, pct_profit, exit_reason FROM papertrade_signals_closure "
        "WHERE signal_id = ?",
        ("sig_exhausted",)
    )
    row = cursor.fetchone()
    assert row is not None
    resolved, pct_profit, exit_reason = row
    assert resolved == 0
    assert pct_profit is None
    assert exit_reason is None

    conn.close()


def test_compute_closures_for_window_engine_version_mismatch_recomputes():
    """engine_version mismatch forces recompute: calling
    compute_closures_for_window twice over the same window with different
    engine_version values should recompute on the second call, and the
    stored closure row should end up stamped with the newer version."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)
    _insert_papertrade_signal(
        conn, signal_id="sig_version", as_of="2026-08-05",
        entry=100.0, stop=95.0, target=105.0, direction="long",
    )

    # A single bar that immediately gaps through target on open -- resolves
    # in one bar so the version-mismatch behavior, not the walk logic, is
    # what's under test here.
    dates = pd.date_range(start="2026-08-05T00:00:00", periods=1, freq="h", tz="UTC")
    bars_df = pd.DataFrame({
        "Open": [106.0],
        "High": [107.0],
        "Low": [105.5],
        "Close": [106.5],
        "Volume": [1000],
    }, index=dates)

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = bars_df
        count_v1 = compute_closures_for_window(
            conn, "2026-08-01", "2026-08-31", interval_ladder=["1h"],
            engine_version="v1", db_path="/tmp/test.db",
        )
    assert count_v1 == 1, f"First call should compute 1 closure, got {count_v1}"

    cursor = conn.execute(
        "SELECT engine_version FROM papertrade_signals_closure WHERE signal_id = ?",
        ("sig_version",)
    )
    assert cursor.fetchone()[0] == "v1"

    # Same window, no new signals -- but a DIFFERENT engine_version should
    # force the existing row to be recomputed.
    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = bars_df
        count_v2 = compute_closures_for_window(
            conn, "2026-08-01", "2026-08-31", interval_ladder=["1h"],
            engine_version="v2", db_path="/tmp/test.db",
        )
    assert count_v2 == 1, (
        f"engine_version mismatch should trigger recompute, got count={count_v2}"
    )

    cursor = conn.execute(
        "SELECT engine_version FROM papertrade_signals_closure WHERE signal_id = ?",
        ("sig_version",)
    )
    assert cursor.fetchone()[0] == "v2", "Closure row should now be stamped with v2"

    conn.close()


def test_compute_closures_for_window_default_db_path_no_crash():
    """Regression test for the E7-S19 class of bug: an explicit db_path=None
    forwarded straight into price_cache.get_price_data overrides its own
    real default and crashes deep inside price_cache._get_conn. Exercises
    compute_closures_for_window's default (db_path omitted) against an
    in-memory DB with zero matching papertrade_signals rows, so the loop
    body never actually reaches price_cache.get_price_data -- but the
    function's own default-resolution logic still runs for real, unmocked.
    """
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    count = compute_closures_for_window(
        conn, "2026-01-01", "2026-01-31", interval_ladder=["1h"],
        engine_version="v1",
        # db_path intentionally omitted
    )

    assert count == 0
    conn.close()


def test_compute_closures_for_window_default_db_path_resolves_to_real_string():
    """Verifies the resolved db_path actually passed to price_cache is
    price_cache.DB_PATH (a real string), never a literal None -- the precise
    mechanism behind the E7-S19 bug class, checked directly rather than only
    inferred from the no-crash smoke test above."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)
    _insert_papertrade_signal(
        conn, signal_id="sig_dbpath", as_of="2026-08-01",
        entry=100.0, stop=95.0, target=105.0, direction="long",
    )

    dates = pd.date_range(start="2026-08-01T00:00:00", periods=1, freq="h", tz="UTC")
    bars_df = pd.DataFrame({
        "Open": [106.0], "High": [107.0], "Low": [105.5], "Close": [106.5],
        "Volume": [1000],
    }, index=dates)

    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = bars_df
        compute_closures_for_window(
            conn, "2026-08-01", "2026-08-31", interval_ladder=["1h"],
            engine_version="v1",
            # db_path intentionally omitted -- must resolve to price_cache.DB_PATH
        )

    assert mock_get.call_count > 0, "Expected price_cache.get_price_data to be called"
    for call in mock_get.call_args_list:
        called_db_path = call.kwargs.get("db_path")
        assert called_db_path is not None, "db_path must not be forwarded as None"
        assert called_db_path == price_cache.DB_PATH

    conn.close()


# ==============================================================================
# Tests for replay_steps and load_step_candidates (E8-S22)
# ==============================================================================


def test_replay_steps_returns_sorted_distinct_as_of():
    """Verify replay_steps returns SORTED, DISTINCT as_of values for an interval
    within a date range (inclusive on both ends)."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # Insert signals at various as_of values
    as_of_values = ["2026-08-05", "2026-08-03", "2026-08-07", "2026-08-05", "2026-08-04"]
    for i, as_of in enumerate(as_of_values):
        conn.execute(
            """
            INSERT INTO papertrade_signals (
                signal_id, strategy_name, ticker, direction, interval, as_of,
                entry, stop, target, expected_value, base_win_rate, n,
                model_label, checkpoint_fingerprint, source_cache_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"sig_{i}", "test_strategy", "AAPL", "long", "1d", as_of,
             100.0, 95.0, 105.0, None, None, None, "base", "", None,
             "2026-08-01T00:00:00Z")
        )
    conn.commit()

    result = replay_steps(conn, "1d", "2026-08-03", "2026-08-07")

    # Should return sorted, distinct values in the range
    expected = ["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-07"]
    assert result == expected, (
        f"Expected {expected}, got {result}"
    )

    conn.close()


def test_replay_steps_respects_interval_filter():
    """Verify replay_steps only returns as_of values for the specified interval."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # Insert signals at different intervals
    conn.execute(
        """
        INSERT INTO papertrade_signals (
            signal_id, strategy_name, ticker, direction, interval, as_of,
            entry, stop, target, expected_value, base_win_rate, n,
            model_label, checkpoint_fingerprint, source_cache_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("sig_1h", "test_strategy", "AAPL", "long", "1h", "2026-08-05T10:00:00",
         100.0, 95.0, 105.0, None, None, None, "base", "", None,
         "2026-08-01T00:00:00Z")
    )
    conn.execute(
        """
        INSERT INTO papertrade_signals (
            signal_id, strategy_name, ticker, direction, interval, as_of,
            entry, stop, target, expected_value, base_win_rate, n,
            model_label, checkpoint_fingerprint, source_cache_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("sig_1d", "test_strategy", "BTC-USD", "short", "1d", "2026-08-05",
         50000.0, 49000.0, 51000.0, None, None, None, "base", "", None,
         "2026-08-01T00:00:00Z")
    )
    conn.commit()

    # Query only 1d interval
    result_1d = replay_steps(conn, "1d", "2026-08-01", "2026-08-31")
    assert result_1d == ["2026-08-05"], (
        f"Expected ['2026-08-05'] for 1d, got {result_1d}"
    )

    # Query only 1h interval
    result_1h = replay_steps(conn, "1h", "2026-08-01", "2026-08-31")
    assert result_1h == ["2026-08-05T10:00:00"], (
        f"Expected ['2026-08-05T10:00:00'] for 1h, got {result_1h}"
    )

    conn.close()


def test_replay_steps_non_daily_spaced_timestamps():
    """Test that replay_steps returns EXACTLY the non-daily-spaced timestamps
    present in the data, proving the step grid is data-driven and not a
    synthesized daily range.

    This is a key test from the acceptance criteria: signals at 1h interval
    with timestamps a few hours apart, not calendar-day-spaced. The function
    should return exactly those timestamps in sorted order."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # Insert 1h signals at non-daily-spaced times
    as_of_values = [
        "2026-08-01T09:00:00",
        "2026-08-01T14:00:00",
        "2026-08-02T03:00:00",
    ]
    for i, as_of in enumerate(as_of_values):
        conn.execute(
            """
            INSERT INTO papertrade_signals (
                signal_id, strategy_name, ticker, direction, interval, as_of,
                entry, stop, target, expected_value, base_win_rate, n,
                model_label, checkpoint_fingerprint, source_cache_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"sig_1h_{i}", "test_strategy", "AAPL", "long", "1h", as_of,
             100.0, 95.0, 105.0, None, None, None, "base", "", None,
             "2026-08-01T00:00:00Z")
        )
    conn.commit()

    result = replay_steps(conn, "1h", "2026-08-01T00:00:00", "2026-08-03T00:00:00")

    # Should return exactly the as_of values present, in sorted order
    assert result == as_of_values, (
        f"Expected {as_of_values}, got {result}"
    )

    conn.close()


def test_load_step_candidates_only_resolved_signals():
    """Test that load_step_candidates returns ONLY resolved signals (resolved=1),
    excluding unresolved (resolved=0) and signals with no closure row at all."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # Insert signals with different resolution states
    signal_ids = ["sig_resolved1", "sig_resolved2", "sig_unresolved", "sig_no_closure"]

    for i, sig_id in enumerate(signal_ids):
        conn.execute(
            """
            INSERT INTO papertrade_signals (
                signal_id, strategy_name, ticker, direction, interval, as_of,
                entry, stop, target, expected_value, base_win_rate, n,
                model_label, checkpoint_fingerprint, source_cache_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sig_id, "test_strategy", f"TICKER{i}", "long", "1d", "2026-08-05",
             100.0, 95.0, 105.0, 1.5, 0.6, 10, "base", "", None,
             "2026-08-01T00:00:00Z")
        )

    # Add closure rows for some
    # sig_resolved1: resolved=1
    conn.execute(
        """
        INSERT INTO papertrade_signals_closure (
            signal_id, resolved, interval_used, pct_profit, max_drawdown_pct,
            trigger_datetime, exit_datetime, exit_reason, computed_at, engine_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("sig_resolved1", 1, "1d", 5.2, 2.1, "2026-08-05T00:00:00Z",
         "2026-08-06T00:00:00Z", "target", "2026-08-01T00:00:00Z", "v1")
    )

    # sig_resolved2: resolved=1
    conn.execute(
        """
        INSERT INTO papertrade_signals_closure (
            signal_id, resolved, interval_used, pct_profit, max_drawdown_pct,
            trigger_datetime, exit_datetime, exit_reason, computed_at, engine_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("sig_resolved2", 1, "1d", 3.1, 1.5, "2026-08-05T00:00:00Z",
         "2026-08-07T00:00:00Z", "target", "2026-08-01T00:00:00Z", "v1")
    )

    # sig_unresolved: resolved=0
    conn.execute(
        """
        INSERT INTO papertrade_signals_closure (
            signal_id, resolved, interval_used, pct_profit, max_drawdown_pct,
            trigger_datetime, exit_datetime, exit_reason, computed_at, engine_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("sig_unresolved", 0, None, None, None, None, None, None, "2026-08-01T00:00:00Z", "v1")
    )

    # sig_no_closure: no closure row at all

    conn.commit()

    result = load_step_candidates(conn, "1d", "2026-08-05")

    # Should return exactly 2 pairs (the resolved ones)
    assert len(result) == 2, (
        f"Expected 2 resolved candidates, got {len(result)}"
    )

    # Verify the returned candidates are the resolved ones
    tickers = [stats_row["symbol"] for stats_row, _ in result]
    assert "TICKER0" in tickers, "sig_resolved1 should be included"
    assert "TICKER1" in tickers, "sig_resolved2 should be included"
    assert "TICKER2" not in tickers, "sig_unresolved should not be included"
    assert "TICKER3" not in tickers, "sig_no_closure should not be included"

    conn.close()


def test_load_step_candidates_dict_structure_and_values():
    """Test that load_step_candidates returns dicts with the correct structure
    and values for feeding into fetch_signals."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # Insert one signal with known values
    conn.execute(
        """
        INSERT INTO papertrade_signals (
            signal_id, strategy_name, ticker, direction, interval, as_of,
            entry, stop, target, expected_value, base_win_rate, n,
            model_label, checkpoint_fingerprint, source_cache_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("sig_test", "test_strategy", "AAPL", "short", "1d", "2026-08-05",
         100.0, 105.0, 95.0, 2.5, 0.65, 15, "base", "", None,
         "2026-08-01T00:00:00Z")
    )

    conn.execute(
        """
        INSERT INTO papertrade_signals_closure (
            signal_id, resolved, interval_used, pct_profit, max_drawdown_pct,
            trigger_datetime, exit_datetime, exit_reason, computed_at, engine_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("sig_test", 1, "1d", 3.2, 1.8, "2026-08-05T00:00:00Z",
         "2026-08-06T12:00:00Z", "target", "2026-08-01T00:00:00Z", "v1")
    )

    conn.commit()

    result = load_step_candidates(conn, "1d", "2026-08-05")

    assert len(result) == 1
    stats_row, advice_row = result[0]

    # Verify stats_row structure
    assert "strategy" in stats_row
    assert "symbol" in stats_row
    assert "direction" in stats_row
    assert "entry" in stats_row
    assert "stop" in stats_row
    assert "target" in stats_row
    assert "expected_value" in stats_row
    assert "base_win_rate" in stats_row

    # Verify stats_row values
    assert stats_row["strategy"] == "test_strategy"
    assert stats_row["symbol"] == "AAPL"
    assert stats_row["direction"] == "short"
    assert stats_row["entry"] == 100.0
    assert stats_row["stop"] == 105.0
    assert stats_row["target"] == 95.0
    assert stats_row["expected_value"] == 2.5
    assert stats_row["base_win_rate"] == 0.65

    # Verify advice_row structure
    assert "expected_value" in advice_row
    assert "entry" in advice_row
    assert "base_win_rate" in advice_row
    assert "base_signals" in advice_row
    assert "oracle_signals" in advice_row
    assert "signal" in advice_row

    # Verify advice_row values
    assert advice_row["expected_value"] == 2.5
    assert advice_row["entry"] == 100.0
    assert advice_row["base_win_rate"] == 0.65
    assert advice_row["base_signals"] == 15
    assert advice_row["oracle_signals"] is None
    assert "AAPL" in advice_row["signal"]
    assert "short" in advice_row["signal"]

    conn.close()


def test_load_step_candidates_integration_with_fetch_signals():
    """Integration test: verify that load_step_candidates' output dicts work
    correctly when passed to strategy/allocation.py's fetch_signals function.

    This confirms the dict structure is exactly what fetch_signals expects."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # Insert a few signals with different characteristics
    signals = [
        {
            "signal_id": "sig_long_1",
            "strategy_name": "TestStrategy1",
            "ticker": "AAPL",
            "direction": "long",
            "interval": "1d",
            "as_of": "2026-08-05",
            "entry": 150.0,
            "stop": 145.0,
            "target": 160.0,
            "expected_value": 3.5,
            "base_win_rate": 0.62,
            "n": 20,
        },
        {
            "signal_id": "sig_short_1",
            "strategy_name": "TestStrategy2",
            "ticker": "MSFT",
            "direction": "short",
            "interval": "1d",
            "as_of": "2026-08-05",
            "entry": 400.0,
            "stop": 410.0,
            "target": 390.0,
            "expected_value": 2.1,
            "base_win_rate": 0.55,
            "n": 12,
        },
    ]

    for sig in signals:
        conn.execute(
            """
            INSERT INTO papertrade_signals (
                signal_id, strategy_name, ticker, direction, interval, as_of,
                entry, stop, target, expected_value, base_win_rate, n,
                model_label, checkpoint_fingerprint, source_cache_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sig["signal_id"], sig["strategy_name"], sig["ticker"],
             sig["direction"], sig["interval"], sig["as_of"],
             sig["entry"], sig["stop"], sig["target"],
             sig["expected_value"], sig["base_win_rate"], sig["n"],
             "base", "", None, "2026-08-01T00:00:00Z")
        )

        # Add resolved closure row
        conn.execute(
            """
            INSERT INTO papertrade_signals_closure (
                signal_id, resolved, interval_used, pct_profit, max_drawdown_pct,
                trigger_datetime, exit_datetime, exit_reason, computed_at, engine_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sig["signal_id"], 1, "1d", 2.0, 1.5, "2026-08-05T00:00:00Z",
             "2026-08-06T00:00:00Z", "target", "2026-08-01T00:00:00Z", "v1")
        )

    conn.commit()

    # Load candidates
    result = load_step_candidates(conn, "1d", "2026-08-05")

    assert len(result) == 2, f"Expected 2 candidates, got {len(result)}"

    # Extract stats_rows and advice_rows
    stats_rows = [stats_row for stats_row, _ in result]
    advice_rows = [advice_row for _, advice_row in result]

    # Import fetch_signals from allocation
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))
    from allocation import fetch_signals

    # Call fetch_signals with the loaded candidates
    # This is the integration check: if the dict structure is wrong, fetch_signals
    # will fail or produce incorrect results.
    candidates = fetch_signals(stats_rows, advice_rows)

    # Verify candidates were successfully created
    assert len(candidates) == 2, (
        f"fetch_signals should produce 2 Candidate objects, got {len(candidates)}"
    )

    # Verify first candidate (long AAPL)
    c1 = candidates[0]
    assert c1.ticker == "AAPL"
    assert c1.direction == "long"
    assert c1.entry == 150.0
    assert c1.stop == 145.0
    assert c1.target == 160.0
    assert c1.base_win_rate == 0.62
    assert c1.n == 20
    assert c1.strategy == "TestStrategy1"

    # Verify second candidate (short MSFT)
    c2 = candidates[1]
    assert c2.ticker == "MSFT"
    assert c2.direction == "short"
    assert c2.entry == 400.0
    assert c2.stop == 410.0
    assert c2.target == 390.0
    assert c2.base_win_rate == 0.55
    assert c2.n == 12
    assert c2.strategy == "TestStrategy2"

    conn.close()


# ==============================================================================
# Tests for replay() (E8-S23)
# ==============================================================================


def _insert_replay_signal(
    conn,
    signal_id: str,
    strategy: str,
    ticker: str,
    direction: str,
    interval: str,
    as_of: str,
    entry: float,
    stop: float,
    target: float,
    ev_pct: float,
    base_win_rate: float,
    n: int,
    pct_profit: float,
) -> None:
    """Helper: insert one fully-resolved (papertrade_signals +
    papertrade_signals_closure) row pair for replay() tests.

    expected_value is stored as an ABSOLUTE value (not a percentage) since
    allocation.fetch_signals() recovers ev_pct via
    _ev_pct_value(expected_value, entry) = expected_value / entry * 100.0 --
    so expected_value = ev_pct * entry / 100.0 reproduces the desired ev_pct
    exactly.
    """
    conn.execute(
        """
        INSERT INTO papertrade_signals (
            signal_id, strategy_name, ticker, direction, interval, as_of,
            entry, stop, target, expected_value, base_win_rate, n,
            model_label, checkpoint_fingerprint, source_cache_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (signal_id, strategy, ticker, direction, interval, as_of,
         entry, stop, target, ev_pct * entry / 100.0, base_win_rate, n,
         "base", "", None, "2026-08-01T00:00:00Z")
    )
    conn.execute(
        """
        INSERT INTO papertrade_signals_closure (
            signal_id, resolved, interval_used, pct_profit, max_drawdown_pct,
            trigger_datetime, exit_datetime, exit_reason, computed_at, engine_version
        ) VALUES (?, 1, ?, ?, 0.0, ?, ?, 'target', '2026-08-01T00:00:00Z', 'v1')
        """,
        (signal_id, interval, pct_profit, as_of, as_of)
    )
    conn.commit()


def test_replay_hand_derived_two_step_scenario():
    """Hand-derived total_profit_eur/num_trades/pct_max_drawdown across 2
    replay steps, each with exactly one (gating-surviving) candidate, using
    a plain default AllocationConfig().

    Step 1 (2026-08-01), candidate AAA: entry=100, stop=90, target=130 ->
    risk_pct=10.0, reward_pct=30.0. avg_win/avg_loss are always None from
    this replay path, so compute_derived's geometry fallback applies:
    b = reward_pct/risk_pct = 3.0, loss_pct = risk_pct = 10.0.
    base_win_rate=0.6, n=100, AllocationConfig defaults n0=100 ->
    shrink = 100/(100+100) = 0.5.
    p_shrunk = 0.5 + (0.6-0.5)*0.5 = 0.55.
    kelly_raw = 0.55 - (1-0.55)/3.0 = 0.55 - 0.15 = 0.40.
    kelly_frac = max(0.40, 0)*kelly_mult(0.35) = 0.14 -> alloc_raw =
    min(14.0, max_pos_pct=15) = 14.0 (not capped).
    ev_pct=5.0 -> ev_shrunk = 5.0*0.5 = 2.5; ev_net = 2.5 -
    round_trip_cost_pct(0.15) = 2.35 > 0 (passes NEG_EV_NET gate); n=100 >=
    min_n=50 (passes LOW_N gate). Sole candidate at this step -> SELECTED,
    alloc=14.0.
    starting_capital=1000.0 -> step_start_capital=1000.0.
    notional = 14.0/100 * 1000.0 = 140.0.
    closure pct_profit=10.0 (a +10% outcome) -> cash_delta =
    140.0 * (10.0/100.0) = 14.0. capital: 1000.0 -> 1014.0.

    Step 2 (2026-08-02), candidate BBB: entry=200, stop=180, target=260 ->
    same risk_pct=10.0/reward_pct=30.0/base_win_rate=0.6/n=100/ev_pct=5.0 as
    step 1, so identically alloc=14.0 (same math as above, price-scale
    invariant).
    step_start_capital = 1014.0 (running capital from step 1).
    notional = 14.0/100 * 1014.0 = 141.96.
    closure pct_profit=-20.0 (a -20% outcome) -> cash_delta =
    141.96 * (-20.0/100.0) = -28.392. capital: 1014.0 -> 985.608.

    Final: total_profit_eur = 985.608 - 1000.0 = -14.392.
    pct_profit = -14.392/1000.0*100.0 = -1.4392.
    num_trades = 2.
    Equity curve: [1000.0, 1014.0, 985.608]. Peak-to-trough drawdown:
    peak tracks 1000.0 -> 1014.0 (equity exceeds peak); at 985.608,
    dd = (1014.0-985.608)/1014.0*100.0 = 28.392/1014.0*100.0 = 2.8
    (exact, since 1014.0*0.028 == 28.392). pct_max_drawdown = 2.8.
    """
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    _insert_replay_signal(
        conn, signal_id="sig_aaa", strategy="S1", ticker="AAA",
        direction="long", interval="1d", as_of="2026-08-01",
        entry=100.0, stop=90.0, target=130.0,
        ev_pct=5.0, base_win_rate=0.6, n=100, pct_profit=10.0,
    )
    _insert_replay_signal(
        conn, signal_id="sig_bbb", strategy="S1", ticker="BBB",
        direction="long", interval="1d", as_of="2026-08-02",
        entry=200.0, stop=180.0, target=260.0,
        ev_pct=5.0, base_win_rate=0.6, n=100, pct_profit=-20.0,
    )

    config = AllocationConfig()
    result = replay(
        conn, interval="1d", start_ts="2026-08-01", end_ts="2026-08-02",
        alloc_config=config, starting_capital=1000.0,
    )

    assert result["total_profit_eur"] == pytest.approx(-14.392)
    assert result["pct_profit"] == pytest.approx(-1.4392)
    assert result["num_trades"] == 2
    assert result["pct_max_drawdown"] == pytest.approx(2.8)

    conn.close()


def test_replay_respects_allocate_top_k_cap():
    """More candidates than alloc_config.top_k at one step -- proves the
    replay loop only applies the pct_profit of the row allocate() actually
    marked SELECTED, not every candidate at that step.

    Candidate AAA: entry=100, stop=90, target=140 -> risk_pct=10.0,
    reward_pct=40.0, b=4.0 (geometry fallback), base_win_rate=0.6, n=100 ->
    shrink=0.5, p_shrunk=0.55.
    kelly_raw = 0.55 - 0.45/4.0 = 0.55 - 0.1125 = 0.4375.
    kelly_frac = 0.4375*0.35 = 0.153125 -> alloc_raw = min(15.3125,
    max_pos_pct=15) = 15.0 (POS_CAPPED, doesn't affect this test).
    ev_pct=8.0 -> ev_net = 8.0*0.5 - 0.15 = 3.85; score = 3.85/10.0 = 0.385.

    Candidate BBB: entry=50, stop=45, target=60 -> risk_pct=10.0,
    reward_pct=20.0, b=2.0, base_win_rate=0.6, n=100 -> same shrink/p_shrunk.
    ev_pct=2.0 -> ev_net = 2.0*0.5 - 0.15 = 0.85; score = 0.85/10.0 = 0.085.

    AAA's score (0.385) > BBB's score (0.085), so with top_k=1 only AAA
    survives ("SELECTED"); BBB is rejected as BELOW_TOPK. BBB's closure
    pct_profit is set to a huge -50.0 -- if the replay loop wrongly applied
    it (bypassing allocate()'s cap), num_trades would be 2 and the profit
    would be strongly negative instead of the small positive value below.

    Expected (AAA only): step_start_capital=1000.0, notional =
    15.0/100*1000.0 = 150.0, pct_profit=5.0 -> cash_delta =
    150.0*(5.0/100.0) = 7.5. total_profit_eur=7.5, num_trades=1.
    """
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    _insert_replay_signal(
        conn, signal_id="sig_aaa_topk", strategy="S1", ticker="AAA",
        direction="long", interval="1d", as_of="2026-08-01",
        entry=100.0, stop=90.0, target=140.0,
        ev_pct=8.0, base_win_rate=0.6, n=100, pct_profit=5.0,
    )
    _insert_replay_signal(
        conn, signal_id="sig_bbb_topk", strategy="S1", ticker="BBB",
        direction="long", interval="1d", as_of="2026-08-01",
        entry=50.0, stop=45.0, target=60.0,
        ev_pct=2.0, base_win_rate=0.6, n=100, pct_profit=-50.0,
    )

    config = AllocationConfig(top_k=1)
    result = replay(
        conn, interval="1d", start_ts="2026-08-01", end_ts="2026-08-01",
        alloc_config=config, starting_capital=1000.0,
    )

    assert result["num_trades"] == 1, (
        "Only the top_k=1 SELECTED candidate (AAA) should have traded; "
        "BBB's pct_profit must not have been applied"
    )
    assert result["total_profit_eur"] == pytest.approx(7.5)
    assert result["pct_profit"] == pytest.approx(0.75)

    conn.close()


def test_replay_rejects_leveraged_config():
    """replay() must fail loudly (not silently) when alloc_config.max_leverage
    is anything above the unleveraged-only invariant of 1.0."""
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    config = AllocationConfig(max_leverage=2.0)

    with pytest.raises(AssertionError):
        replay(
            conn, interval="1d", start_ts="2026-08-01", end_ts="2026-08-02",
            alloc_config=config, starting_capital=1000.0,
        )

    conn.close()


# ==============================================================================
# Tests for _build_arg_parser (CLI wiring)
# ==============================================================================


class TestBuildArgParser:
    """Tests for the CLI argument parser."""

    def test_parser_accepts_precompute_flag(self):
        """Verify parser accepts --precompute flag."""
        args = _build_arg_parser().parse_args([
            "--precompute", "--start", "2026-08-01", "--end", "2026-08-07"
        ])
        assert args.precompute is True
        assert args.replay is False

    def test_parser_accepts_replay_flag(self):
        """Verify parser accepts --replay flag."""
        args = _build_arg_parser().parse_args([
            "--replay", "--start", "2026-08-01", "--end", "2026-08-07",
            "--interval", "1d", "--capital", "200"
        ])
        assert args.replay is True
        assert args.precompute is False

    def test_parser_start_end_required(self):
        """Verify --start and --end are required for both modes."""
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--precompute"])

    def test_parser_interval_ladder_default(self):
        """Verify --interval-ladder defaults to '1h,4h,1d'."""
        args = _build_arg_parser().parse_args([
            "--precompute", "--start", "2026-08-01", "--end", "2026-08-07"
        ])
        assert args.interval_ladder == "1h,4h,1d"

    def test_parser_interval_ladder_custom(self):
        """Verify --interval-ladder accepts custom comma-separated values."""
        args = _build_arg_parser().parse_args([
            "--precompute", "--start", "2026-08-01", "--end", "2026-08-07",
            "--interval-ladder", "1h,1d"
        ])
        assert args.interval_ladder == "1h,1d"

    def test_parser_engine_version_default(self):
        """Verify --engine-version defaults to 'v1'."""
        args = _build_arg_parser().parse_args([
            "--precompute", "--start", "2026-08-01", "--end", "2026-08-07"
        ])
        assert args.engine_version == "v1"

    def test_parser_engine_version_custom(self):
        """Verify --engine-version accepts custom value."""
        args = _build_arg_parser().parse_args([
            "--precompute", "--start", "2026-08-01", "--end", "2026-08-07",
            "--engine-version", "v2"
        ])
        assert args.engine_version == "v2"

    def test_parser_db_default_is_SIGNALS_DB_PATH(self):
        """Verify --db defaults to kairos_signals.DB_PATH."""
        args = _build_arg_parser().parse_args([
            "--precompute", "--start", "2026-08-01", "--end", "2026-08-07"
        ])
        from kairos_signals import DB_PATH as SIGNALS_DB_PATH
        assert args.db == SIGNALS_DB_PATH

    def test_parser_db_custom(self):
        """Verify --db accepts custom path."""
        custom_path = "/tmp/custom.db"
        args = _build_arg_parser().parse_args([
            "--precompute", "--start", "2026-08-01", "--end", "2026-08-07",
            "--db", custom_path
        ])
        assert args.db == custom_path

    def test_parser_replay_capital_float(self):
        """Verify --capital accepts float values."""
        args = _build_arg_parser().parse_args([
            "--replay", "--start", "2026-08-01", "--end", "2026-08-07",
            "--interval", "1d", "--capital", "500.5"
        ])
        assert args.capital == 500.5

    def test_parser_replay_max_pos_pct_default(self):
        """Verify --max-pos-pct defaults to 15.0."""
        args = _build_arg_parser().parse_args([
            "--replay", "--start", "2026-08-01", "--end", "2026-08-07",
            "--interval", "1d", "--capital", "200"
        ])
        assert args.max_pos_pct == 15.0

    def test_parser_replay_max_pos_pct_custom(self):
        """Verify --max-pos-pct accepts custom value."""
        args = _build_arg_parser().parse_args([
            "--replay", "--start", "2026-08-01", "--end", "2026-08-07",
            "--interval", "1d", "--capital", "200", "--max-pos-pct", "10"
        ])
        assert args.max_pos_pct == 10.0

    def test_parser_replay_top_k_default(self):
        """Verify --top-k defaults to 12."""
        args = _build_arg_parser().parse_args([
            "--replay", "--start", "2026-08-01", "--end", "2026-08-07",
            "--interval", "1d", "--capital", "200"
        ])
        assert args.top_k == 12

    def test_parser_replay_top_k_custom(self):
        """Verify --top-k accepts custom integer value."""
        args = _build_arg_parser().parse_args([
            "--replay", "--start", "2026-08-01", "--end", "2026-08-07",
            "--interval", "1d", "--capital", "200", "--top-k", "5"
        ])
        assert args.top_k == 5

    def test_parser_replay_signal_selection_default_none(self):
        """Verify --signal-selection defaults to None."""
        args = _build_arg_parser().parse_args([
            "--replay", "--start", "2026-08-01", "--end", "2026-08-07",
            "--interval", "1d", "--capital", "200"
        ])
        assert args.signal_selection is None

    def test_parser_replay_signal_selection_custom(self):
        """Verify --signal-selection accepts a rule string."""
        rule = "'n' > 60, 'Win raw' > 0.6, ORDER 'EV raw %' DESC, TOP 3"
        args = _build_arg_parser().parse_args([
            "--replay", "--start", "2026-08-01", "--end", "2026-08-07",
            "--interval", "1d", "--capital", "200",
            "--signal-selection", rule
        ])
        assert args.signal_selection == rule

    def test_parser_interval_none_by_default(self):
        """Verify --interval defaults to None."""
        args = _build_arg_parser().parse_args([
            "--precompute", "--start", "2026-08-01", "--end", "2026-08-07"
        ])
        assert args.interval is None

    def test_parser_interval_custom_for_replay(self):
        """Verify --interval accepts custom value for replay mode."""
        args = _build_arg_parser().parse_args([
            "--replay", "--start", "2026-08-01", "--end", "2026-08-07",
            "--interval", "4h", "--capital", "200"
        ])
        assert args.interval == "4h"

    def test_parser_capital_default_none(self):
        """Verify --capital defaults to None (must be provided for replay)."""
        args = _build_arg_parser().parse_args([
            "--precompute", "--start", "2026-08-01", "--end", "2026-08-07"
        ])
        assert args.capital is None


# ==============================================================================
# E9-S25: Dedicated cache-reuse & engine_version-bump regression tests
# ==============================================================================


def test_precompute_is_idempotent():
    """Dedicated idempotency test for the full precompute pipeline.

    Calls BOTH unpack_signals_cache_to_papertrade_signals AND
    compute_closures_for_window twice over the exact same window and data,
    asserting:
    - The SECOND pass inserts/recomputes zero rows (both functions return 0)
    - created_at timestamps on papertrade_signals rows are UNCHANGED between passes
    - computed_at timestamps on papertrade_signals_closure rows are UNCHANGED between passes

    Per DESIGN_DOC_offline_signal_replay.md §5 (Testing plan, "Cache reuse" row):
    "Run --precompute twice over the same window | Second run touches zero new rows,
    confirmed via row-count/`created_at` check"

    This proves the cache-reuse behavior works end-to-end: both the signal
    unpacking and closure computation phases are idempotent.
    """
    conn = sqlite3.connect(":memory:")
    _build_signals_cache_table(conn)
    _ensure_signal_replay_tables(conn)

    # Build synthetic signals_cache row with one LONG signal
    stats_json = json.dumps([
        {
            "strategy": "test_strategy_cache_reuse",
            "symbol": "BTC-USD",
            "direction": "LONG",
            "entry": 50000.0,
            "stop": 49000.0,
            "target": 51000.0,
            "expected_value": 0.8,
            "base_win_rate": 0.62,
        },
    ])

    advice_json = json.dumps([
        {
            "expected_value": 0.8,
            "base_signals": 25,
            "oracle_signals": 20,
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
        ("cache_key_idempotent_1", "test_strategy_cache_reuse", "BTC-USD", "1d",
         "2026-08-07", 100, 50, 0.0, "base", None, "", stats_json,
         advice_json, "[]", "2026-08-07T10:00:00Z")
    )
    conn.commit()

    # Create synthetic bars for closure computation
    dates = pd.date_range(start="2026-08-07T00:00:00", periods=3, freq="h", tz="UTC")
    bars_df = pd.DataFrame({
        "Open": [50100.0, 50300.0, 50800.0],
        "High": [50200.0, 50500.0, 51100.0],
        "Low": [50000.0, 50200.0, 50700.0],
        "Close": [50150.0, 50400.0, 51050.0],
        "Volume": [100, 100, 100],
    }, index=dates)

    # ===== FIRST PASS =====
    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = bars_df

        unpacked_first = unpack_signals_cache_to_papertrade_signals(
            conn, "2026-08-01", "2026-08-31"
        )
        assert unpacked_first == 1, f"First unpack should insert 1 row, got {unpacked_first}"

        computed_first = compute_closures_for_window(
            conn, "2026-08-01", "2026-08-31", interval_ladder=["1h"],
            engine_version="v1", db_path="/tmp/test.db",
        )
        assert computed_first == 1, f"First compute should process 1 signal, got {computed_first}"

    # Capture timestamps after first pass
    cursor = conn.execute(
        "SELECT signal_id, created_at FROM papertrade_signals WHERE as_of = '2026-08-07'"
    )
    signal_first_pass = cursor.fetchone()
    assert signal_first_pass is not None, "Signal should exist after first unpack"
    signal_id, created_at_first = signal_first_pass

    cursor = conn.execute(
        "SELECT computed_at, engine_version FROM papertrade_signals_closure WHERE signal_id = ?",
        (signal_id,)
    )
    closure_first_pass = cursor.fetchone()
    assert closure_first_pass is not None, "Closure should exist after first compute"
    computed_at_first, engine_version_first = closure_first_pass
    assert engine_version_first == "v1"

    # ===== SECOND PASS (idempotent, same window) =====
    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = bars_df

        unpacked_second = unpack_signals_cache_to_papertrade_signals(
            conn, "2026-08-01", "2026-08-31"
        )
        assert unpacked_second == 0, \
            f"Second unpack over same window should insert 0 rows, got {unpacked_second}"

        computed_second = compute_closures_for_window(
            conn, "2026-08-01", "2026-08-31", interval_ladder=["1h"],
            engine_version="v1", db_path="/tmp/test.db",
        )
        assert computed_second == 0, \
            f"Second compute with same engine_version should compute 0 rows, got {computed_second}"

    # Verify timestamps are UNCHANGED
    cursor = conn.execute(
        "SELECT created_at FROM papertrade_signals WHERE signal_id = ?",
        (signal_id,)
    )
    created_at_second = cursor.fetchone()[0]
    assert created_at_second == created_at_first, \
        f"created_at should be unchanged: first={created_at_first}, second={created_at_second}"

    cursor = conn.execute(
        "SELECT computed_at, engine_version FROM papertrade_signals_closure WHERE signal_id = ?",
        (signal_id,)
    )
    computed_at_second, engine_version_second = cursor.fetchone()
    assert computed_at_second == computed_at_first, \
        f"computed_at should be unchanged: first={computed_at_first}, second={computed_at_second}"
    assert engine_version_second == "v1", "engine_version should remain v1"

    conn.close()


def test_engine_version_bump_forces_recompute():
    """Dedicated test for engine_version cache invalidation.

    Calls compute_closures_for_window twice with the same window but different
    engine_version values, asserting:
    - First call (engine_version="v1") computes closure rows
    - Second call (engine_version="v2") recomputes EXISTING rows (count > 0)
    - computed_at timestamp reflects the new computation time
    - engine_version column is updated to "v2"

    Per DESIGN_DOC_offline_signal_replay.md §5 (Testing plan, "engine_version
    bump forces recompute" row): "Bump the constant, rerun --precompute |
    Existing closure rows get recomputed, not silently left stale"

    This proves that an engine_version mismatch properly invalidates and
    recomputes cached closure rows, matching the design's cache-busting
    pattern (similar to signals_cache/kairos_predcache's checkpoint_fingerprint).
    """
    conn = sqlite3.connect(":memory:")
    _ensure_signal_replay_tables(conn)

    # Insert a papertrade_signals row directly (bypass unpacking)
    signal_id = "sig_engine_version_test"
    conn.execute(
        """
        INSERT INTO papertrade_signals (
            signal_id, strategy_name, ticker, direction, interval, as_of,
            entry, stop, target, expected_value, base_win_rate, n,
            model_label, checkpoint_fingerprint, source_cache_key, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (signal_id, "test_strategy", "ETH-USD", "long", "1h", "2026-08-08T12:00:00",
         3000.0, 2900.0, 3100.0, 1.2, 0.65, 30, "base", "", None,
         "2026-08-08T00:00:00Z")
    )
    conn.commit()

    # Create synthetic bars for closure computation
    dates = pd.date_range(start="2026-08-08T12:00:00", periods=2, freq="h", tz="UTC")
    bars_df = pd.DataFrame({
        "Open": [3010.0, 3040.0],
        "High": [3020.0, 3110.0],
        "Low": [3005.0, 3035.0],
        "Close": [3015.0, 3105.0],
        "Volume": [50, 50],
    }, index=dates)

    # ===== FIRST CALL: engine_version="v1" =====
    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = bars_df
        count_v1 = compute_closures_for_window(
            conn, "2026-08-01", "2026-08-31", interval_ladder=["1h"],
            engine_version="v1", db_path="/tmp/test.db",
        )

    assert count_v1 == 1, f"First call should compute 1 closure, got {count_v1}"

    # Capture v1 closure state
    cursor = conn.execute(
        "SELECT resolved, engine_version, computed_at FROM papertrade_signals_closure "
        "WHERE signal_id = ?",
        (signal_id,)
    )
    closure_v1 = cursor.fetchone()
    assert closure_v1 is not None, "Closure row should exist after v1 computation"
    resolved_v1, engine_version_v1, computed_at_v1 = closure_v1
    assert resolved_v1 == 1, "Closure should be resolved (not disqualified)"
    assert engine_version_v1 == "v1"
    assert computed_at_v1 is not None

    # Sleep briefly to ensure computed_at timestamp differs (and to separate wall-clock time)
    import time
    time.sleep(0.01)

    # ===== SECOND CALL: engine_version="v2" (forces recompute) =====
    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = bars_df
        count_v2 = compute_closures_for_window(
            conn, "2026-08-01", "2026-08-31", interval_ladder=["1h"],
            engine_version="v2", db_path="/tmp/test.db",
        )

    # This is the key assertion: engine_version mismatch should force recompute
    assert count_v2 == 1, (
        f"engine_version bump should recompute existing row, got count={count_v2} "
        "(expected 1 since there's 1 signal needing recompute)"
    )

    # Capture v2 closure state
    cursor = conn.execute(
        "SELECT resolved, engine_version, computed_at FROM papertrade_signals_closure "
        "WHERE signal_id = ?",
        (signal_id,)
    )
    closure_v2 = cursor.fetchone()
    assert closure_v2 is not None, "Closure row should still exist after v2 recompute"
    resolved_v2, engine_version_v2, computed_at_v2 = closure_v2
    assert resolved_v2 == 1, "Closure should still be resolved"
    assert engine_version_v2 == "v2", \
        f"engine_version should be updated to v2, got {engine_version_v2}"
    assert computed_at_v2 is not None, "computed_at should be set"
    assert computed_at_v2 != computed_at_v1, \
        f"computed_at should change on recompute: v1={computed_at_v1}, v2={computed_at_v2}"

    # ===== THIRD CALL: engine_version="v2" again (should be idempotent) =====
    with patch("kairos_signal_replay.price_cache.get_price_data") as mock_get:
        mock_get.return_value = bars_df
        count_v2_again = compute_closures_for_window(
            conn, "2026-08-01", "2026-08-31", interval_ladder=["1h"],
            engine_version="v2", db_path="/tmp/test.db",
        )

    assert count_v2_again == 0, (
        f"Calling with same engine_version again should be idempotent, "
        f"got count={count_v2_again} (expected 0)"
    )

    # Verify timestamp hasn't changed
    cursor = conn.execute(
        "SELECT computed_at FROM papertrade_signals_closure WHERE signal_id = ?",
        (signal_id,)
    )
    computed_at_v2_again = cursor.fetchone()[0]
    assert computed_at_v2_again == computed_at_v2, (
        f"computed_at should not change when version matches: "
        f"expected {computed_at_v2}, got {computed_at_v2_again}"
    )

    conn.close()
