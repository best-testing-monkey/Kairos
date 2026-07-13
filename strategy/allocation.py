"""allocation.py — Candidate schema and fetch_signals adapter for portfolio allocation.

Adapts the existing signals-report row data (stats_rows, advice_rows from kairos_signals.py)
into structured Candidate objects matching RFC allocation_sheet.md §3.
"""

import math
from dataclasses import dataclass
from typing import Optional

from kairos_signals import _ev_pct_value


@dataclass
class Candidate:
    """Structured representation of a single trade candidate for allocation.

    Maps one-to-one with a non-FLAT signal from kairos_signals.py run().
    Fields match RFC allocation_sheet.md §3 required schema, plus nullable fields.
    """
    strategy: str
    ticker: str
    direction: str  # "long" or "short" (FLAT excluded)
    entry: float
    stop: float
    target: float
    ev_pct: float
    base_win_rate: float
    n: int
    backtest_period: str
    sharpe: float
    advised_liquidity_pct: float
    avg_win_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None
    avg_holding_days: Optional[float] = None


def validate_candidate(c: Candidate) -> Optional[str]:
    """Validate a Candidate against schema constraints per RFC allocation_sheet.md §3.

    Returns "SCHEMA_ERROR" if:
      - Any required field is None
      - Any numeric required field (entry, stop, target, ev_pct, base_win_rate, sharpe)
        is non-finite (NaN or inf)
      - Direction/stop/target placement is inconsistent with direction:
        * direction == "long" requires stop < entry < target
        * direction == "short" requires target < entry < stop

    Returns None if the candidate is valid.

    Args:
        c: Candidate object to validate

    Returns:
        "SCHEMA_ERROR" if invalid, None if valid
    """
    # Check all required fields are not None
    if (c.strategy is None or c.ticker is None or c.direction is None or
        c.entry is None or c.stop is None or c.target is None or
        c.ev_pct is None or c.base_win_rate is None or c.n is None or
        c.backtest_period is None or c.sharpe is None):
        return "SCHEMA_ERROR"

    # Check that numeric fields are finite (not NaN or inf)
    numeric_fields = [c.entry, c.stop, c.target, c.ev_pct, c.base_win_rate, c.sharpe]
    for field in numeric_fields:
        if not math.isfinite(field):
            return "SCHEMA_ERROR"

    # Check direction/stop/target placement consistency
    if c.direction == "long":
        # For long: stop < entry < target
        if not (c.stop < c.entry < c.target):
            return "SCHEMA_ERROR"
    elif c.direction == "short":
        # For short: target < entry < stop
        if not (c.target < c.entry < c.stop):
            return "SCHEMA_ERROR"
    else:
        # Invalid direction value (not "long" or "short")
        return "SCHEMA_ERROR"

    return None


def fetch_signals(stats_rows, advice_rows):
    """Adapt stats_rows and advice_rows from kairos_signals.py run() into Candidate objects.

    Args:
        stats_rows: list of dicts with keys from kairos_signals.py STATS_COLUMNS
                    (strategy, symbol, direction, entry, stop, target, expected_value,
                     base_sharpe, base_win_rate, etc.)
        advice_rows: list of dicts with keys (expected_value, entry, base_win_rate,
                     base_signals, oracle_signals, signal)

    Returns:
        list of Candidate objects, one per non-FLAT stats_row, correctly paired
        to its corresponding advice_row by list index (both lists are built in lockstep
        in kairos_signals.py run()).

    Exclusions:
        - Rows with direction == "FLAT" are excluded entirely
        - Stats rows and advice rows are matched by index, so both lists must be
          the same length and built in the same order
    """
    candidates = []

    for stats_row, advice_row in zip(stats_rows, advice_rows):
        # Exclude FLAT direction rows
        direction_str = stats_row.get("direction", "").upper()
        if direction_str == "FLAT":
            continue

        # Normalize direction to lowercase "long" or "short"
        direction = direction_str.lower()

        # Extract ev_pct using the same helper as kairos_signals.py
        ev_pct = _ev_pct_value(
            stats_row.get("expected_value"),
            stats_row.get("entry")
        )
        # If ev_pct could not be computed, skip this row
        # (though in practice this should not happen for non-FLAT signals)
        if ev_pct is None:
            continue

        # Fallback for n: use base_signals, then oracle_signals, then None
        # (mirroring build_signals_table's fallback in kairos_signals.py:290-293)
        n_value = None
        base_signals = advice_row.get("base_signals")
        oracle_signals = advice_row.get("oracle_signals")

        if base_signals is not None and not _is_missing(base_signals):
            n_value = int(base_signals)
        elif oracle_signals is not None and not _is_missing(oracle_signals):
            n_value = int(oracle_signals)

        # Skip if n could not be determined
        if n_value is None:
            continue

        # Compute advised_liquidity_pct from size
        size = stats_row.get("size", 0.0)
        advised_liquidity_pct = size * 100.0 if size else 0.0

        # Construct the Candidate
        candidate = Candidate(
            strategy=stats_row.get("strategy", ""),
            ticker=stats_row.get("symbol", ""),
            direction=direction,
            entry=float(stats_row.get("entry", 0.0)),
            stop=float(stats_row.get("stop", 0.0)),
            target=float(stats_row.get("target", 0.0)),
            ev_pct=ev_pct,
            base_win_rate=float(stats_row.get("base_win_rate", 0.0)),
            n=n_value,
            backtest_period=str(stats_row.get("backtest_period", "")),
            sharpe=float(stats_row.get("base_sharpe", 0.0)),
            advised_liquidity_pct=advised_liquidity_pct,
            avg_win_pct=None,  # Not present in current data; v1 nullable
            avg_loss_pct=None,  # Not present in current data; v1 nullable
            avg_holding_days=None,  # Not present in current data; v1 nullable
        )

        candidates.append(candidate)

    return candidates


def _is_missing(value):
    """Check if a value is missing (None or NaN).

    Replicates the helper from kairos_signals.py for consistency.
    """
    import numpy as np

    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False
