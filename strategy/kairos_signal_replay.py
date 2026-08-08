#!/usr/bin/env python3
"""kairos_signal_replay.py — Offline signal replay and allocation testing.

Fast, offline tool for replaying and testing selection and allocation rules
against precomputed signal outcomes. No GPU, no live phantom execution, no
model inference — closure statistics are computed once from historical price
bars, then the replay loop runs arbitrary allocation configs against those
precomputed outcomes in seconds instead of hours.

See DESIGN_DOC_offline_signal_replay.md for full design, schema, and testing
plan.
"""

import hashlib
import json
from datetime import datetime, timezone, timedelta

import price_cache  # type: ignore


def _ensure_signal_replay_tables(conn) -> None:
    """Create the `papertrade_signals` and `papertrade_signals_closure` tables
    in whatever DB `conn` points at, if they don't already exist. Creates
    both tables and the `idx_papertrade_signals_as_of` index. Safe to call
    multiple times (uses CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT
    EXISTS).

    Schema per DESIGN_DOC_offline_signal_replay.md Section 3.1:
    - papertrade_signals: one row per individual signal, normalized from
      signals_cache's per-group JSON blobs
    - papertrade_signals_closure: one row per signal, its isolated outcome
      (entry, exit, PnL, max drawdown) when traded alone
    - idx_papertrade_signals_as_of: index on as_of for fast replay-window
      lookups
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papertrade_signals (
            signal_id               TEXT PRIMARY KEY,
            strategy_name           TEXT NOT NULL,
            ticker                  TEXT NOT NULL,
            direction               TEXT NOT NULL,
            interval                TEXT NOT NULL,
            as_of                   TEXT NOT NULL,
            entry                   REAL NOT NULL,
            stop                    REAL NOT NULL,
            target                  REAL NOT NULL,
            expected_value          REAL,
            base_win_rate           REAL,
            n                       INTEGER,
            model_label             TEXT NOT NULL,
            checkpoint_fingerprint  TEXT NOT NULL DEFAULT '',
            source_cache_key        TEXT,
            created_at              TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_papertrade_signals_as_of
        ON papertrade_signals(as_of)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS papertrade_signals_closure (
            signal_id           TEXT PRIMARY KEY
                                REFERENCES papertrade_signals(signal_id),
            resolved            INTEGER NOT NULL,
            interval_used       TEXT,
            pct_profit          REAL,
            max_drawdown_pct    REAL,
            trigger_datetime    TEXT,
            exit_datetime       TEXT,
            exit_reason         TEXT,
            computed_at         TEXT NOT NULL,
            engine_version      TEXT NOT NULL
        )
        """
    )
    conn.commit()


def unpack_signals_cache_to_papertrade_signals(conn, start_date: str, end_date: str) -> int:
    """Unpack signals_cache rows into individual papertrade_signals rows.

    Reads signals_cache entries with as_of in [start_date, end_date] (inclusive,
    string comparison), parses their stats_json and advice_json, and inserts one
    papertrade_signals row per non-FLAT (stats_row, advice_row) pair matched by
    list index. Uses INSERT OR IGNORE keyed on signal_id to ensure idempotent
    re-runs return 0 new rows.

    Args:
        conn: sqlite3 connection to pipeline_results.db
        start_date: ISO date string (e.g., "2026-08-01"), inclusive
        end_date: ISO date string (e.g., "2026-08-07"), inclusive

    Returns:
        Count of rows actually inserted (0 if re-run over same window)
    """
    _ensure_signal_replay_tables(conn)

    # Track inserted rows
    initial_changes = conn.total_changes

    # Query signals_cache for the date range
    cursor = conn.execute(
        """
        SELECT cache_key, strategy_name, interval, as_of, model_label,
               checkpoint_fingerprint, stats_json, advice_json
        FROM signals_cache
        WHERE as_of >= ? AND as_of <= ?
        ORDER BY as_of
        """,
        (start_date, end_date)
    )

    rows = cursor.fetchall()

    for (cache_key, strategy_name, interval, as_of, model_label,
         checkpoint_fingerprint, stats_json_str, advice_json_str) in rows:

        # Parse JSON
        try:
            stats_list = json.loads(stats_json_str)
            advice_list = json.loads(advice_json_str)
        except (json.JSONDecodeError, ValueError):
            # Skip rows with malformed JSON
            continue

        # Zip by index; skip if lists differ in length (shouldn't happen, but safe)
        for stats_row, advice_row in zip(stats_list, advice_list):
            # Skip FLAT direction
            direction_str = stats_row.get("direction", "").upper()
            if direction_str == "FLAT":
                continue

            # Normalize direction to lowercase
            direction = direction_str.lower()

            # Extract required fields from stats_row
            ticker = stats_row.get("symbol")
            entry = stats_row.get("entry")
            stop = stats_row.get("stop")
            target = stats_row.get("target")

            # Skip if required fields are missing
            if ticker is None or entry is None or stop is None or target is None:
                continue

            # Cast numeric fields
            try:
                entry = float(entry)
                stop = float(stop)
                target = float(target)
            except (ValueError, TypeError):
                continue

            # Optional fields
            expected_value = stats_row.get("expected_value")
            if expected_value is not None:
                try:
                    expected_value = float(expected_value)
                except (ValueError, TypeError):
                    expected_value = None

            base_win_rate = stats_row.get("base_win_rate")
            if base_win_rate is not None:
                try:
                    base_win_rate = float(base_win_rate)
                except (ValueError, TypeError):
                    base_win_rate = None

            # Fallback for n: base_signals, then oracle_signals
            n_value = None
            base_signals = advice_row.get("base_signals")
            oracle_signals = advice_row.get("oracle_signals")

            if base_signals is not None:
                try:
                    n_value = int(base_signals)
                except (ValueError, TypeError):
                    pass

            if n_value is None and oracle_signals is not None:
                try:
                    n_value = int(oracle_signals)
                except (ValueError, TypeError):
                    pass

            # Build deterministic signal_id hash
            # Format: strategy_name|ticker|direction|interval|as_of|entry|stop|target
            #         |model_label|checkpoint_fingerprint
            signal_parts = [
                strategy_name or "",
                ticker or "",
                direction or "",
                interval or "",
                as_of or "",
                str(entry),
                str(stop),
                str(target),
                model_label or "",
                checkpoint_fingerprint or ""
            ]
            signal_str = "|".join(signal_parts)
            signal_id = hashlib.sha256(signal_str.encode()).hexdigest()

            # Current UTC timestamp in ISO format
            created_at = datetime.now(timezone.utc).isoformat()

            # Insert into papertrade_signals
            conn.execute(
                """
                INSERT OR IGNORE INTO papertrade_signals (
                    signal_id, strategy_name, ticker, direction, interval,
                    as_of, entry, stop, target, expected_value, base_win_rate,
                    n, model_label, checkpoint_fingerprint, source_cache_key,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (signal_id, strategy_name, ticker, direction, interval,
                 as_of, entry, stop, target, expected_value, base_win_rate,
                 n_value, model_label, checkpoint_fingerprint, cache_key,
                 created_at)
            )

    conn.commit()

    # Return count of newly inserted rows
    return conn.total_changes - initial_changes


def resolve_interval_for_signal(
    ticker: str,
    entry_datetime,
    interval_ladder: list[str],
    min_bars: int = 2,
    db_path: str = price_cache.DB_PATH
) -> str | None:
    """Resolve the smallest available interval for a signal's closure computation.

    Tries each interval in interval_ladder (smallest-first, order as provided by
    caller) for the given ticker. Returns the first interval whose price data has
    at least min_bars bars within a forward window from entry_datetime. Returns
    None if no interval succeeds.

    The forward window is 30 days from entry_datetime — sufficient time for most
    trades to hit a stop/target or time out. Intraday intervals (1h, 5m, etc.)
    still resolve within this window; daily intervals step through dates as usual.

    Does not raise on price_cache.get_price_data returning None, raising an
    exception, or returning an empty DataFrame. Instead, silently moves to the
    next interval in the ladder.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL")
        entry_datetime: Entry timestamp (datetime object) — used as start of
            forward window
        interval_ladder: List of interval strings to try, in order
            (e.g. ["1h", "4h", "1d"]). Caller is responsible for sorting
            smallest-first
        min_bars: Minimum bars required for a successful interval (default 2)
        db_path: Path to price_cache SQLite database

    Returns:
        The interval string from interval_ladder that succeeded, or None if all
        failed
    """
    if not interval_ladder or not ticker:
        return None

    # Forward window: 30 days from entry_datetime
    # This gives enough time for most trades to resolve (hit stop/target/timeout)
    forward_days = 30

    # Convert entry_datetime to date strings for price_cache
    if hasattr(entry_datetime, 'date'):
        start_date = entry_datetime.date().isoformat()
    else:
        start_date = entry_datetime.isoformat()

    end_datetime = entry_datetime + timedelta(days=forward_days)
    if hasattr(end_datetime, 'date'):
        end_date = end_datetime.date().isoformat()
    else:
        end_date = end_datetime.isoformat()

    for interval in interval_ladder:
        try:
            df = price_cache.get_price_data(
                ticker,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                db_path=db_path
            )
        except Exception:
            # price_cache call failed; try next interval
            continue

        if df is None or df.empty or len(df) < min_bars:
            # Not enough bars for this interval; try next
            continue

        # Success — return the first interval with sufficient data
        return interval

    # No interval succeeded
    return None
