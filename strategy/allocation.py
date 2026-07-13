"""allocation.py — Candidate schema and fetch_signals adapter for portfolio allocation.

Adapts the existing signals-report row data (stats_rows, advice_rows from kairos_signals.py)
into structured Candidate objects matching RFC allocation_sheet.md §3.

Implements AllocationConfig (RFC §3.1 defaults) and compute_derived (RFC §4.2 per-row formulas).
"""

import math
from dataclasses import dataclass, field
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


@dataclass
class AllocationConfig:
    """Configuration for portfolio allocation sizing and gating.

    Per RFC allocation_sheet.md §3.1, defaults are deliberately round numbers
    swept in Phantom Ledger. Do not ship precise-looking fitted values.
    """
    n0: int = 100  # Shrinkage constant; weight of the "no edge" prior
    min_n: int = 50  # Reject signals with fewer backtest trades
    round_trip_cost_pct: float = 0.15  # Assumed total cost per round trip
    kelly_mult: float = 0.35  # Fractional Kelly multiplier
    top_k: int = 12  # Max number of positions
    max_pos_pct: float = 15  # Cap per position, % of equity
    max_cluster_pct: float = 25  # Cap per correlation cluster, % of equity
    gross_cap_pct: float = 100  # Total gross exposure cap
    dust_min_pct: float = 1.0  # Zero out final allocations below this, % of equity
    equity: Optional[float] = None  # Optional account equity for currency-amount column
    cluster_map: dict = field(default_factory=dict)  # ticker -> cluster name, static mapping


def compute_ev_ratio(c: Candidate, derived: dict) -> tuple[float, bool]:
    """Compute ev_ratio and DATA_MISMATCH flag per RFC allocation_sheet.md §4.3.

    Implements the data-quality check comparing empirical EV (from backtest) against
    EV implied by geometry (base_win_rate, risk_pct, reward_pct). Flags significant
    divergence.

    Args:
        c: Candidate object with ev_pct and base_win_rate
        derived: dict from compute_derived() with keys risk_pct, reward_pct

    Returns:
        (ev_ratio, is_mismatch) tuple where:
        - ev_ratio: float, the ratio ev_pct / ev_implied (or 0.0 if ev_implied near zero)
        - is_mismatch: bool, True iff ev_ratio is outside [0.5, 2.0] and ev_implied is not near zero.
                       If ev_implied is near zero (< 1e-9 in absolute value), treat as not-mismatched
                       since the ratio is undefined, not a data problem.
    """
    risk_pct = derived["risk_pct"]
    reward_pct = derived["reward_pct"]

    # Compute ev_implied per RFC §4.3
    ev_implied = c.base_win_rate * reward_pct - (1 - c.base_win_rate) * risk_pct

    # Guard against near-zero denominator
    if abs(ev_implied) < 1e-9:
        # Not mismatched when ev_implied is too small to define a meaningful ratio
        return 0.0, False

    # Compute ev_ratio
    ev_ratio = c.ev_pct / ev_implied

    # Check if ratio is outside the band [0.5, 2.0]
    is_mismatch = ev_ratio < 0.5 or ev_ratio > 2.0

    return ev_ratio, is_mismatch


def compute_derived(c: Candidate, config: AllocationConfig) -> dict:
    """Compute per-row derived columns per RFC allocation_sheet.md §4.2.

    Implements the exact formulas for derived allocation metrics, including the two branches:
    - Empirical branch: when avg_win_pct and avg_loss_pct are both present
    - Geometry-fallback branch: when either is None, use TP/SL geometry

    Args:
        c: Candidate object with all required fields
        config: AllocationConfig with n0, round_trip_cost_pct, kelly_mult

    Returns:
        dict with keys: risk_pct, reward_pct, b, loss_pct, shrink, ev_shrunk,
        ev_net, p_shrunk, kelly_raw, kelly_frac, score

    Note: No division-by-zero checks needed. risk_pct and loss_pct are guaranteed
    positive (stop/target placement validated in schema), and shrink is in [0,1).
    """
    # Per-row derived columns, per RFC §4.2
    risk_pct = abs(c.stop - c.entry) / c.entry * 100
    reward_pct = abs(c.target - c.entry) / c.entry * 100

    # Payoff ratio: empirical when available, geometry as fallback
    if c.avg_win_pct is not None and c.avg_loss_pct is not None:
        b = c.avg_win_pct / c.avg_loss_pct
        loss_pct = c.avg_loss_pct
    else:
        b = reward_pct / risk_pct
        loss_pct = risk_pct

    shrink = c.n / (c.n + config.n0)
    ev_shrunk = c.ev_pct * shrink
    ev_net = ev_shrunk - config.round_trip_cost_pct

    p_shrunk = 0.5 + (c.base_win_rate - 0.5) * shrink
    kelly_raw = p_shrunk - (1 - p_shrunk) / b
    kelly_frac = max(kelly_raw, 0) * config.kelly_mult

    score = ev_net / loss_pct

    return {
        "risk_pct": risk_pct,
        "reward_pct": reward_pct,
        "b": b,
        "loss_pct": loss_pct,
        "shrink": shrink,
        "ev_shrunk": ev_shrunk,
        "ev_net": ev_net,
        "p_shrunk": p_shrunk,
        "kelly_raw": kelly_raw,
        "kelly_frac": kelly_frac,
        "score": score,
    }


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


def select_candidates(candidates: list[Candidate], config: AllocationConfig, enabled_mask: dict) -> list[dict]:
    """Select and reject candidates through gating, collapse, and top-K ranking.

    Implements RFC §4.4 selection algorithm: gate → per-asset collapse → rank+top-K.

    Rejection reasons (per RFC §4.5):
    - SCHEMA_ERROR: required field missing or direction/stop/target inconsistent
    - DISABLED: enabled_mask[ticker] is False
    - LOW_N: n < config.min_n
    - NEG_EV_NET: ev_net <= 0
    - DIRECTION_CONFLICT: both long and short survive gating for same ticker
    - DUP_ASSET: duplicate ticker/direction, not the max-score row
    - BELOW_TOPK: survivor after collapse, but outside top config.top_k

    Args:
        candidates: list of Candidate objects
        config: AllocationConfig with gating/ranking parameters
        enabled_mask: dict mapping ticker -> bool; get(ticker, True) defaults to enabled

    Returns:
        list of dicts (one per candidate), in input order, each containing:
        - All original Candidate fields (strategy, ticker, direction, entry, stop, target, etc.)
        - derived: dict with keys from compute_derived (risk_pct, reward_pct, score, etc.)
        - status: rejection reason (str) or None for rows proceeding to E11-S06 sizing
        - flags: list starting with ["DATA_MISMATCH"] if ev_ratio flagged, else []
    """
    from collections import defaultdict

    # Convert all candidates to dicts and prepare for processing
    output_rows = {}  # original_index -> row dict

    for i, c in enumerate(candidates):
        row = _candidate_to_dict(c)
        output_rows[i] = row

    # =========================================================================
    # Stage 1: GATE
    # =========================================================================
    # Gate order (RFC §4.4): SCHEMA_ERROR → DISABLED → LOW_N → NEG_EV_NET
    # First matching reason wins; survivors get status=None
    for i, c in enumerate(candidates):
        row = output_rows[i]

        # Check SCHEMA_ERROR first (earliest in gate order)
        if validate_candidate(c) is not None:
            row["status"] = "SCHEMA_ERROR"
            row["flags"] = []
            row["derived"] = {}  # No derived for invalid schema
            continue

        # Compute derived fields (needed for ev_net and later stages)
        derived = compute_derived(c, config)
        row["derived"] = derived

        # Compute ev_ratio and DATA_MISMATCH flag
        ev_ratio, is_mismatch = compute_ev_ratio(c, derived)
        row["flags"] = ["DATA_MISMATCH"] if is_mismatch else []

        # Check DISABLED (second in gate order)
        if not enabled_mask.get(c.ticker, True):
            row["status"] = "DISABLED"
            continue

        # Check LOW_N (third in gate order)
        if c.n < config.min_n:
            row["status"] = "LOW_N"
            continue

        # Check NEG_EV_NET (fourth in gate order)
        if derived["ev_net"] <= 0:
            row["status"] = "NEG_EV_NET"
            continue

        # Survived gating
        row["status"] = None

    # =========================================================================
    # Stage 2: PER-ASSET COLLAPSE (on survivors only)
    # =========================================================================
    # Group survivors by ticker; check for direction conflict or mark duplicates
    ticker_groups = defaultdict(list)

    for i, c in enumerate(candidates):
        row = output_rows[i]
        if row["status"] is not None:
            continue  # Skip rejected rows
        ticker_groups[c.ticker].append((i, c, row))

    # For each ticker, check direction conflict or mark duplicates
    for ticker, group in ticker_groups.items():
        directions = set(c.direction for _, c, _ in group)

        if len(directions) > 1:
            # Both long and short survived gating for this ticker
            # Reject all rows for that ticker with DIRECTION_CONFLICT
            for _, _, row in group:
                row["status"] = "DIRECTION_CONFLICT"
        else:
            # Only one direction; keep max-score row, reject rest as DUP_ASSET
            if len(group) > 1:
                # Find index of max-score row
                max_idx = max(
                    range(len(group)),
                    key=lambda j: group[j][2]["derived"]["score"]
                )
                for j, (_, _, row) in enumerate(group):
                    if j != max_idx:
                        row["status"] = "DUP_ASSET"

    # =========================================================================
    # Stage 3: RANK + TOP-K (on survivors only)
    # =========================================================================
    # Collect survivors after collapse, maintaining original index order
    survivors = []
    for i, c in enumerate(candidates):
        row = output_rows[i]
        if row["status"] is None:
            survivors.append((i, row, c))

    # Sort by score descending (stable sort preserves insertion order on ties)
    # per RFC §4.4's deterministic tie-break note
    survivors.sort(key=lambda x: x[1]["derived"]["score"], reverse=True)

    # Mark top-K as selected (status remains None), rest as BELOW_TOPK
    for rank, (_, row, _) in enumerate(survivors):
        if rank >= config.top_k:
            row["status"] = "BELOW_TOPK"

    # Return all rows in original input order (not sorted)
    return [output_rows[i] for i in range(len(candidates))]


def _candidate_to_dict(c: Candidate) -> dict:
    """Convert Candidate dataclass to dict with all fields.

    Returns a dict ready for output, with all Candidate fields plus placeholders
    for derived, status, and flags fields to be filled in by select_candidates().
    """
    return {
        "strategy": c.strategy,
        "ticker": c.ticker,
        "direction": c.direction,
        "entry": c.entry,
        "stop": c.stop,
        "target": c.target,
        "ev_pct": c.ev_pct,
        "base_win_rate": c.base_win_rate,
        "n": c.n,
        "backtest_period": c.backtest_period,
        "sharpe": c.sharpe,
        "advised_liquidity_pct": c.advised_liquidity_pct,
        "avg_win_pct": c.avg_win_pct,
        "avg_loss_pct": c.avg_loss_pct,
        "avg_holding_days": c.avg_holding_days,
        "derived": {},  # Filled in by select_candidates
        "status": None,  # Filled in by select_candidates
        "flags": [],  # Filled in by select_candidates
    }


def size_selected(survivors: list[dict], config: AllocationConfig) -> list[dict]:
    """Apply position cap, cluster caps, gross cap, and dust filter to top-K survivors.

    Implements RFC §4.4 sizing pipeline (position cap → cluster caps → gross cap → dust filter)
    on the survivors from select_candidates(). Returns the full row set (all candidates, both
    selected and rejected) with final sizing applied to rows with status=None (top-K).

    Args:
        survivors: Full output list from select_candidates(), containing both rejected rows
                  (with status set) and top-K survivors (with status=None)
        config: AllocationConfig with sizing parameters (max_pos_pct, max_cluster_pct,
               gross_cap_pct, dust_min_pct, cluster_map)

    Returns:
        List of dicts with the same rows as input, each top-K survivor now having:
        - alloc: final allocation % after all caps and dust filter
        - status: "SELECTED" or "DUST" (for survivors); unchanged for rejected rows
        - flags: may include "POS_CAPPED" and/or "CLUSTER_CAPPED" (for survivors)
        Rejected rows pass through unchanged.

    Per RFC §4.6, dust filter is single-pass: zeroed allocations are not redistributed.
    """
    from collections import defaultdict

    # Deep copy to avoid mutating input
    result = [dict(row) for row in survivors]

    # =========================================================================
    # Stage 1: POSITION CAP (per-row, on survivors only)
    # =========================================================================
    # alloc = min(kelly_frac * 100, max_pos_pct)
    for row in result:
        if row["status"] is not None:
            # Rejected row; skip sizing
            continue

        kelly_frac = row["derived"]["kelly_frac"]
        alloc_raw = min(kelly_frac * 100, config.max_pos_pct)

        # Flag if capped
        if kelly_frac * 100 > config.max_pos_pct:
            if "POS_CAPPED" not in row["flags"]:
                row["flags"].append("POS_CAPPED")

        row["alloc"] = alloc_raw

    # =========================================================================
    # Stage 2: CLUSTER CAPS (proportional scale-down within over-cap clusters)
    # =========================================================================
    # Group post-cap allocations by cluster; if cluster sum > max_cluster_pct,
    # scale that cluster's allocations proportionally
    cluster_groups = defaultdict(list)
    for row in result:
        if row["status"] is not None:
            # Rejected row; skip
            continue

        ticker = row["ticker"]
        # Unmapped tickers form their own singleton cluster (their own ticker
        # name as cluster key), so unrelated unmapped tickers never get
        # cluster-capped together under one shared bucket.
        cluster = config.cluster_map.get(ticker, ticker)
        cluster_groups[cluster].append(row)

    for cluster, cluster_rows in cluster_groups.items():
        cluster_sum = sum(row["alloc"] for row in cluster_rows)

        if cluster_sum > config.max_cluster_pct:
            # Scale factor: new_sum / old_sum
            scale_factor = config.max_cluster_pct / cluster_sum

            for row in cluster_rows:
                row["alloc"] *= scale_factor

                # Add CLUSTER_CAPPED flag if not already present
                if "CLUSTER_CAPPED" not in row["flags"]:
                    row["flags"].append("CLUSTER_CAPPED")

    # =========================================================================
    # Stage 3: GROSS CAP (proportional scale-down if total > gross_cap_pct)
    # =========================================================================
    survivors_with_status_none = [row for row in result if row["status"] is None]
    gross_sum = sum(row["alloc"] for row in survivors_with_status_none)

    if gross_sum > config.gross_cap_pct:
        scale_factor = config.gross_cap_pct / gross_sum

        for row in survivors_with_status_none:
            row["alloc"] *= scale_factor

    # =========================================================================
    # Stage 4: DUST FILTER (single-pass, no redistribution)
    # =========================================================================
    # Any row with final alloc < dust_min_pct gets alloc=0 and status="DUST"
    for row in result:
        if row["status"] is not None:
            # Rejected row; skip
            continue

        if row["alloc"] < config.dust_min_pct:
            row["alloc"] = 0.0
            row["status"] = "DUST"
        else:
            row["status"] = "SELECTED"

    return result


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
