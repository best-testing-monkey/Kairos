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
