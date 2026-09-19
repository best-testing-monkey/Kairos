#!/usr/bin/env python3
"""Compute grid-candidate hindsight labels for historical signals.

For each precomputed historical signal in papertrade_signals, resolves a small
grid of alternative (stop_pct, target_pct) candidates against that signal's
real subsequent price path, and persists a binary "did target hit before stop"
label per (signal, candidate) in tpsl_label_candidates table.

Price-path-only; no model/GPU access needed. Idempotent (INSERT OR REPLACE on
the natural key), safe to re-run.

Usage:
    uv run scripts/tpsl_label_grid.py [--db PATH] [--limit N]

The candidate grid is: STOP_PCT_GRID = [10.0, 15.0, 20.0],
TARGET_PCT_GRID = [75.0, 85.0, 90.0] (9 combinations per signal).
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd
import price_cache  # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "strategy"))

from kairos_backtest import BacktestEngine, Direction  # noqa: E402, type: ignore
from kairos_signal_replay import (  # noqa: E402, type: ignore
    _ensure_configured_db,
    resolve_interval_for_signal,
    _TERMINAL_EXIT_REASONS,
    _CLOSURE_FORWARD_DAYS,
)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "pipeline_results.db")

# Candidate grid (hardcoded as module constants)
STOP_PCT_GRID = [10.0, 15.0, 20.0]
TARGET_PCT_GRID = [75.0, 85.0, 90.0]

# Engine version for cache-busting (bump when candidate logic changes)
ENGINE_VERSION = "e18_s01_v1"


def _ensure_tpsl_label_candidates_table(conn) -> None:
    """Create the tpsl_label_candidates table if it doesn't exist."""
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


def _compute_candidate_prices(
    entry: float,
    direction: str,
    stop_pct: float,
    target_pct: float,
) -> tuple[float, float]:
    """Compute stop/target prices for a candidate grid entry.

    For LONG: stop is below entry, target is above entry.
    For SHORT: stop is above entry (to limit upward loss), target is below
    entry (to profit from downward move).

    Args:
        entry: Entry price
        direction: 'long' or 'short'
        stop_pct: Stop loss percentage
        target_pct: Take profit percentage

    Returns:
        Tuple of (candidate_stop, candidate_target)
    """
    if direction.lower() == "long":
        candidate_stop = entry * (1.0 - stop_pct / 100.0)
        candidate_target = entry * (1.0 + target_pct / 100.0)
    else:  # SHORT
        candidate_stop = entry * (1.0 + stop_pct / 100.0)
        candidate_target = entry * (1.0 - target_pct / 100.0)

    return candidate_stop, candidate_target


def _write_disqualified_candidate(
    conn,
    signal_id: str,
    stop_pct: float,
    target_pct: float,
    computed_at: str,
) -> None:
    """Write a resolved=0 tpsl_label_candidates row for a candidate."""
    conn.execute(
        """
        INSERT OR REPLACE INTO tpsl_label_candidates (
            signal_id, stop_pct, target_pct, resolved, hit_target_first,
            interval_used, engine_version, computed_at
        ) VALUES (?, ?, ?, 0, NULL, NULL, ?, ?)
        """,
        (signal_id, stop_pct, target_pct, ENGINE_VERSION, computed_at)
    )


def resolve_candidate(
    conn,
    signal_row: dict[str, Any],
    stop_pct: float,
    target_pct: float,
    interval_ladder: list[str],
    fee_pct: float,
    slippage_pct: float,
    db_path: str,
) -> None:
    """Resolve one candidate against a signal's price path.

    Computes the candidate stop/target prices, walks forward bars to resolve
    exits, and writes one tpsl_label_candidates row (INSERT OR REPLACE).

    Args:
        conn: sqlite3 connection to pipeline_results.db
        signal_row: dict with signal_id, ticker, direction, entry, as_of, etc.
        stop_pct: Candidate stop percentage
        target_pct: Candidate target percentage
        interval_ladder: Intervals to try, smallest-first
        fee_pct: Fee fraction
        slippage_pct: Slippage fraction
        db_path: price_cache database path
    """
    signal_id = signal_row["signal_id"]
    ticker = signal_row["ticker"]
    direction_str = str(signal_row["direction"]).lower()
    entry = float(signal_row["entry"])
    entry_datetime = pd.to_datetime(signal_row["as_of"])

    computed_at = datetime.now(timezone.utc).isoformat()

    # Ensure price_cache is configured for local mode before any price lookup
    _ensure_configured_db(db_path)

    # Compute candidate stop/target prices
    candidate_stop, candidate_target = _compute_candidate_prices(
        entry, direction_str, stop_pct, target_pct
    )

    # Resolve interval for this signal
    resolved_interval = resolve_interval_for_signal(
        ticker=ticker,
        entry_datetime=entry_datetime,
        interval_ladder=interval_ladder,
        db_path=db_path,
    )
    if resolved_interval is None:
        _write_disqualified_candidate(conn, signal_id, stop_pct, target_pct, computed_at)
        return

    # Fetch forward bars
    end_datetime = entry_datetime + timedelta(days=_CLOSURE_FORWARD_DAYS)
    try:
        bars = price_cache.get_price_data(
            ticker,
            start_date=entry_datetime.date().isoformat(),
            end_date=end_datetime.date().isoformat(),
            interval=resolved_interval,
            db_path=db_path,
        )
    except Exception:
        bars = None

    if bars is None or bars.empty:
        _write_disqualified_candidate(conn, signal_id, stop_pct, target_pct, computed_at)
        return

    direction = Direction.LONG if direction_str == "long" else Direction.SHORT
    position: dict[str, Any] = {
        "direction": direction,
        "entry": entry,
        "stop": candidate_stop,
        "target": candidate_target,
        "size": 1.0,
    }
    engine = BacktestEngine(predictor=None, fee_pct=fee_pct, slippage_pct=slippage_pct)  # type: ignore[arg-type]

    hit_target_first: int | None = None

    for ts, row in bars.iterrows():
        # Rebuild bar with lowercase keys (BacktestEngine._check_exit convention)
        bar = pd.Series({
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
        })
        exit_price, exit_reason = engine._check_exit(position, bar)

        if exit_reason in _TERMINAL_EXIT_REASONS:
            # Determine which barrier was hit
            if exit_reason in ("target", "target_open"):
                hit_target_first = 1
            elif exit_reason in ("stop", "stop_open"):
                hit_target_first = 0
            else:
                # Should not happen for terminal reasons, but be defensive
                hit_target_first = 0

            conn.execute(
                """
                INSERT OR REPLACE INTO tpsl_label_candidates (
                    signal_id, stop_pct, target_pct, resolved, hit_target_first,
                    interval_used, engine_version, computed_at
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                """,
                (
                    signal_id,
                    stop_pct,
                    target_pct,
                    hit_target_first,
                    resolved_interval,
                    ENGINE_VERSION,
                    computed_at,
                )
            )
            conn.commit()
            return

    # No terminal exit found -- disqualify
    _write_disqualified_candidate(conn, signal_id, stop_pct, target_pct, computed_at)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", default=DB_PATH, help="Path to pipeline_results.db")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit signals processed (for testing)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # Create table
    _ensure_tpsl_label_candidates_table(conn)

    # Interval ladder to try, smallest-first
    interval_ladder = ["1h", "4h", "1d"]

    # Cost model (matching kairos_signal_replay defaults)
    fee_pct = 0.001
    slippage_pct = 0.0005

    # Configure price_cache
    _ensure_configured_db(args.db)

    # Query signals not yet fully covered for current engine_version
    cursor = conn.execute(
        """
        SELECT DISTINCT s.signal_id, s.ticker, s.direction, s.entry, s.as_of
        FROM papertrade_signals s
        LEFT JOIN tpsl_label_candidates c
            ON s.signal_id = c.signal_id AND c.engine_version = ?
        WHERE c.signal_id IS NULL
        ORDER BY s.as_of
        """,
        (ENGINE_VERSION,)
    )

    uncovered_signals = cursor.fetchall()
    if args.limit:
        uncovered_signals = uncovered_signals[:args.limit]

    print(f"Found {len(uncovered_signals)} signals not yet covered for {ENGINE_VERSION}")

    total_candidates = 0
    for i, signal_row in enumerate(uncovered_signals):
        signal_id = signal_row["signal_id"]
        if (i + 1) % 100 == 0:
            print(f"  Processing signal {i + 1}/{len(uncovered_signals)}: {signal_id[:8]}...")

        for stop_pct in STOP_PCT_GRID:
            for target_pct in TARGET_PCT_GRID:
                resolve_candidate(
                    conn,
                    signal_row,
                    stop_pct,
                    target_pct,
                    interval_ladder,
                    fee_pct,
                    slippage_pct,
                    args.db,
                )
                total_candidates += 1

    print(f"Processed {total_candidates} candidates across {len(uncovered_signals)} signals")

    # Summary stats
    resolved_count = conn.execute(
        "SELECT COUNT(*) FROM tpsl_label_candidates WHERE resolved=1"
    ).fetchone()[0]
    disqualified_count = conn.execute(
        "SELECT COUNT(*) FROM tpsl_label_candidates WHERE resolved=0"
    ).fetchone()[0]
    target_hits = conn.execute(
        "SELECT COUNT(*) FROM tpsl_label_candidates WHERE hit_target_first=1"
    ).fetchone()[0]
    stop_hits = conn.execute(
        "SELECT COUNT(*) FROM tpsl_label_candidates WHERE hit_target_first=0"
    ).fetchone()[0]

    print("\nSummary:")
    print(f"  Resolved:        {resolved_count}")
    print(f"  Disqualified:    {disqualified_count}")
    print(f"  Target hits:     {target_hits}")
    print(f"  Stop hits:       {stop_hits}")

    conn.close()


if __name__ == "__main__":
    main()
