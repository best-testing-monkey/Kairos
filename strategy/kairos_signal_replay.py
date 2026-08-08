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
from typing import Any

import pandas as pd
import price_cache  # type: ignore

from kairos_backtest import BacktestEngine, Direction

# Forward window used when re-fetching bars for closure computation, matching
# resolve_interval_for_signal's own forward window (kept as a separate literal
# rather than a shared import so this module's two 30-day windows can diverge
# independently if a future story needs that).
_CLOSURE_FORWARD_DAYS = 30

# _check_exit's real contract (confirmed against strategy/kairos_backtest.py's
# current source, ~line 1997) is NOT "returns (None, None) to keep walking" as
# an earlier draft of the design doc assumed -- it ALWAYS returns a resolved
# (price, reason) tuple for whatever single bar it's given, falling back to
# ("close" price, "close") when neither a gap-through nor an intrabar
# stop/target touch occurred. That fallback exists because BacktestEngine.run()
# only ever calls it once, against the very next bar, and forces the position
# closed either way. Closure computation instead has many forward bars
# available, so "close" is treated as NON-terminal here (keep walking to the
# next bar) and only a genuine stop/target/gap exit ends the walk. See
# DESIGN_DOC_offline_signal_replay.md §3.3.
_TERMINAL_EXIT_REASONS = frozenset({"stop_open", "target_open", "stop", "target"})


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


def max_adverse_excursion_pct(direction: str, entry_price: float, bars: pd.DataFrame) -> float:
    """Compute the worst (most adverse) direction-aware % move from entry price.

    Walks each bar in the provided DataFrame and tracks the worst adverse move
    relative to entry price. Returns the magnitude of that worst move as a
    positive percentage (0.0 if never underwater).

    For long positions: adverse move is when price falls (uses bar's Low).
    For short positions: adverse move is when price rises (uses bar's High).

    The returned value reflects the most negative PnL-equivalent move (expressed
    as a percentage of entry price). This matches the direction-aware convention
    from kairos_mtm.unrealized_pnl():
    - long: (price - entry_price) * quantity
    - short: (entry_price - price) * quantity

    Args:
        direction: "long" or "short" (case-sensitive, lowercase required)
        entry_price: Entry price of the position
        bars: DataFrame with at least "Low" and "High" columns (capitalized)

    Returns:
        Worst adverse excursion as a positive percentage, or 0.0 if:
        - bars is empty
        - position was never underwater at any bar

    Raises:
        ValueError: If direction is not "long" or "short"
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    if bars.empty:
        return 0.0

    worst_pct_move = 0.0  # Track the most negative move; start at 0.0

    if direction == "long":
        # For long: adverse move uses Low price
        # pct_move = (Low - entry_price) / entry_price * 100.0 (negative when underwater)
        for low_price in bars["Low"]:
            pct_move = (low_price - entry_price) / entry_price * 100.0
            if pct_move < worst_pct_move:
                worst_pct_move = pct_move
    else:  # direction == "short"
        # For short: adverse move uses High price
        # pct_move = (entry_price - High) / entry_price * 100.0 (negative when underwater)
        for high_price in bars["High"]:
            pct_move = (entry_price - high_price) / entry_price * 100.0
            if pct_move < worst_pct_move:
                worst_pct_move = pct_move

    # Return the magnitude as a positive percentage, or 0.0 if never underwater
    return abs(worst_pct_move) if worst_pct_move < 0.0 else 0.0


def _write_disqualified_closure(conn, signal_id: str, engine_version: str, computed_at: str) -> None:
    """Write a resolved=0 papertrade_signals_closure row for signal_id.

    Per DESIGN_DOC_offline_signal_replay.md §3.1: a disqualified signal is a
    real, present row marked unresolved, not a silently absent one. Every
    column besides signal_id/resolved/engine_version/computed_at is NULL.
    """
    conn.execute(
        """
        INSERT OR REPLACE INTO papertrade_signals_closure (
            signal_id, resolved, interval_used, pct_profit, max_drawdown_pct,
            trigger_datetime, exit_datetime, exit_reason, computed_at, engine_version
        ) VALUES (?, 0, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?)
        """,
        (signal_id, computed_at, engine_version)
    )
    conn.commit()


def compute_closure(
    conn,
    signal_row: dict[str, Any],
    interval_ladder: list[str],
    fee_pct: float,
    slippage_pct: float,
    db_path: str,
    engine_version: str,
) -> None:
    """Resolve and persist one papertrade_signals row's isolated outcome.

    Reuses BacktestEngine's private, predictor-free _check_exit/_calculate_pnl
    primitives (see DESIGN_DOC_offline_signal_replay.md §3.3) to resolve an
    ALREADY-DECIDED signal (entry/stop/target/direction from signal_row)
    against forward price bars -- no prediction/routing step, since the
    decision is already made.

    signal_row must provide: signal_id, ticker, direction ("long"/"short",
    case-insensitive), entry, stop, target, and as_of. papertrade_signals has
    no separate "entry_datetime" column; as_of (the signal-generation
    timestamp, per the table's own schema comment) is parsed as the entry
    datetime and used as the start of the forward bar-fetch window.

    Writes exactly one papertrade_signals_closure row (INSERT OR REPLACE
    keyed on signal_id):
    - resolved=0, every other column NULL (except engine_version/computed_at)
      if resolve_interval_for_signal finds no usable interval, OR if the
      forward-bar walk exhausts every fetched bar without a genuine
      stop/target/gap exit (see _TERMINAL_EXIT_REASONS above for why "close"
      alone does not count as resolving -- both are documented disqualification
      paths, not errors).
    - resolved=1 with interval_used/pct_profit/max_drawdown_pct/
      trigger_datetime/exit_datetime/exit_reason populated otherwise.
      pct_profit and max_drawdown_pct are both percentage NUMBERS on the same
      scale (5.2 means 5.2%, not 0.052) -- consumers (e.g. the E8 replay
      loop) can treat both columns identically.

    Args:
        conn: sqlite3 connection to pipeline_results.db (tables assumed to
            already exist, e.g. via compute_closures_for_window)
        signal_row: mapping with the fields described above (a papertrade_signals
            row, as a dict)
        interval_ladder: intervals to try, smallest-first (passed through to
            resolve_interval_for_signal)
        fee_pct: flat fee fraction of notional, per BacktestEngine's own
            cost convention (NOT phantom's per-broker model -- see §3.3)
        slippage_pct: flat slippage fraction, same convention
        db_path: price_cache SQLite database path
        engine_version: cache-busting tag; stored on the written row so a
            future bump forces recompute (see compute_closures_for_window)
    """
    signal_id = signal_row["signal_id"]
    ticker = signal_row["ticker"]
    direction_str = str(signal_row["direction"]).lower()
    entry = float(signal_row["entry"])
    stop = float(signal_row["stop"])
    target = float(signal_row["target"])
    entry_datetime = pd.to_datetime(signal_row["as_of"])

    computed_at = datetime.now(timezone.utc).isoformat()

    resolved_interval = resolve_interval_for_signal(
        ticker=ticker,
        entry_datetime=entry_datetime,
        interval_ladder=interval_ladder,
        db_path=db_path,
    )
    if resolved_interval is None:
        _write_disqualified_closure(conn, signal_id, engine_version, computed_at)
        return

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
        # resolve_interval_for_signal just confirmed sufficient bars exist at
        # this interval/window; an empty refetch here is unexpected, but the
        # safe response is the same disqualification path, not a crash.
        _write_disqualified_closure(conn, signal_id, engine_version, computed_at)
        return

    direction = Direction.LONG if direction_str == "long" else Direction.SHORT
    position_size = 1.0  # isolated per-signal PnL; pct_profit is size-invariant
    position: dict[str, Any] = {
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "target": target,
        "size": position_size,
    }
    engine = BacktestEngine(predictor=None, fee_pct=fee_pct, slippage_pct=slippage_pct)  # type: ignore[arg-type]

    worst_pct_move = 0.0  # max adverse excursion bookkeeping, same loop, no second pass
    exit_price: float | None = None
    exit_reason: str | None = None
    exit_timestamp: pd.Timestamp | None = None

    for ts, row in bars.iterrows():
        low = float(row["Low"])
        high = float(row["High"])

        if direction == Direction.LONG:
            pct_move = (low - entry) / entry * 100.0
        else:
            pct_move = (entry - high) / entry * 100.0
        if pct_move < worst_pct_move:
            worst_pct_move = pct_move

        # _check_exit reads lowercase open/high/low/close (its own internal
        # convention) -- price_cache returns capitalized Open/High/Low/Close,
        # so the bar is rebuilt with lowercase keys before being handed over.
        bar = pd.Series({
            "open": float(row["Open"]),
            "high": high,
            "low": low,
            "close": float(row["Close"]),
        })
        candidate_price, candidate_reason = engine._check_exit(position, bar)
        if candidate_reason in _TERMINAL_EXIT_REASONS:
            exit_price = candidate_price
            exit_reason = candidate_reason
            exit_timestamp = ts
            break

    if exit_price is None or exit_reason is None or exit_timestamp is None:
        # Bars ran out without a genuine stop/target/gap exit -- disqualified,
        # same path as an interval-resolution failure (documented choice,
        # per this story's acceptance criteria).
        _write_disqualified_closure(conn, signal_id, engine_version, computed_at)
        return

    pnl = engine._calculate_pnl(position, exit_price)
    # Trade.pnl_pct's own convention (BacktestEngine.run(), same file) is a raw
    # fraction (pnl / (entry_price * size)); scaled by 100 here so pct_profit
    # is on the same "percentage number" scale as max_drawdown_pct below (both
    # papertrade_signals_closure columns mean e.g. 5.2 for 5.2%, NOT 0.052 --
    # see the schema note in this function's docstring).
    pct_profit = pnl / (entry * position_size) * 100.0
    max_drawdown_pct = abs(worst_pct_move) if worst_pct_move < 0.0 else 0.0

    conn.execute(
        """
        INSERT OR REPLACE INTO papertrade_signals_closure (
            signal_id, resolved, interval_used, pct_profit, max_drawdown_pct,
            trigger_datetime, exit_datetime, exit_reason, computed_at, engine_version
        ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            signal_id,
            resolved_interval,
            pct_profit,
            max_drawdown_pct,
            entry_datetime.isoformat(),
            exit_timestamp.isoformat(),
            exit_reason,
            computed_at,
            engine_version,
        )
    )
    conn.commit()


def compute_closures_for_window(
    conn,
    start_date: str,
    end_date: str,
    interval_ladder: list[str],
    engine_version: str,
    fee_pct: float = 0.001,
    slippage_pct: float = 0.0005,
    db_path: str | None = None,
) -> int:
    """Compute closures for every papertrade_signals row in [start_date, end_date]
    that needs one: no closure row yet, or its closure row's engine_version
    doesn't match the current engine_version (forces recompute, mirroring
    signals_cache/kairos_predcache's own fingerprint-based invalidation).

    db_path defaults to None here so the common call site doesn't have to name
    price_cache.DB_PATH explicitly, but None is resolved to the real default
    BEFORE being passed to compute_closure/price_cache -- forwarding a literal
    None through would override price_cache.get_price_data's own db_path
    default and crash inside price_cache._get_conn (this is the exact bug
    class E7-S19 hit).

    Args:
        conn: sqlite3 connection to pipeline_results.db
        start_date: ISO date string, inclusive (compared against as_of)
        end_date: ISO date string, inclusive
        interval_ladder: intervals to try per signal, smallest-first
        engine_version: current engine version tag; rows stamped with a
            different (or missing) engine_version are recomputed
        fee_pct: flat fee fraction, passed through to compute_closure
        slippage_pct: flat slippage fraction, passed through
        db_path: price_cache SQLite database path, or None to use
            price_cache.DB_PATH

    Returns:
        Count of papertrade_signals rows for which compute_closure was called
    """
    _ensure_signal_replay_tables(conn)
    resolved_db_path = db_path if db_path is not None else price_cache.DB_PATH

    cursor = conn.execute(
        """
        SELECT s.signal_id, s.ticker, s.direction, s.entry, s.stop, s.target, s.as_of
        FROM papertrade_signals s
        LEFT JOIN papertrade_signals_closure c ON s.signal_id = c.signal_id
        WHERE s.as_of >= ? AND s.as_of <= ?
          AND (c.signal_id IS NULL OR c.engine_version != ?)
        """,
        (start_date, end_date, engine_version)
    )
    rows = cursor.fetchall()

    for signal_id, ticker, direction, entry, stop, target, as_of in rows:
        signal_row = {
            "signal_id": signal_id,
            "ticker": ticker,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "target": target,
            "as_of": as_of,
        }
        compute_closure(
            conn, signal_row, interval_ladder, fee_pct, slippage_pct,
            resolved_db_path, engine_version,
        )

    return len(rows)
