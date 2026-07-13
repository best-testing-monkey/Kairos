"""allocation.py — Candidate schema and fetch_signals adapter for portfolio allocation.

Adapts the existing signals-report row data (stats_rows, advice_rows from kairos_signals.py)
into structured Candidate objects matching RFC allocation_sheet.md §3.

Implements AllocationConfig (RFC §3.1 defaults) and compute_derived (RFC §4.2 per-row formulas).
"""

import csv
import math
from dataclasses import dataclass, field
from typing import Optional

from openpyxl.styles import Protection

from kairos_signals import _ev_pct_value, format_table


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


@dataclass
class AllocationResult:
    """Result of allocate() orchestration per RFC allocation_sheet.md §4.4.

    The top-level output dataclass carrying the full allocation decision, including
    all rows (selected and rejected), summary statistics, and rejection counts.
    """
    rows: list[dict] = field(default_factory=list)  # One per input candidate with all fields + derived/status/flags/alloc
    selected_count: int = 0  # Number of rows with status == "SELECTED"
    gross_exposure_pct: float = 0.0  # Sum of alloc across status == "SELECTED" rows
    rejection_counts: dict[str, int] = field(default_factory=dict)  # status -> count for all non-selected rows


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


def allocate(candidates: list[Candidate], config: AllocationConfig, enabled_mask: dict = None) -> AllocationResult:
    """Top-level allocation orchestration per RFC allocation_sheet.md §8.

    Implements the complete selection and sizing pipeline:
    1. validate_candidate (E11-S02): schema validation
    2. compute_derived (E11-S03): per-row derived metrics
    3. compute_ev_ratio (E11-S04): data quality check
    4. select_candidates (E11-S05): gating, collapse, top-K ranking
    5. size_selected (E11-S06): position/cluster/gross caps + dust filter

    This is the reference oracle per RFC §4 and §8 ("pure function of (candidates, config)").

    Args:
        candidates: list of Candidate objects
        config: AllocationConfig with all sizing/gating parameters
        enabled_mask: dict mapping ticker -> bool; defaults to all-enabled (empty dict
                     since select_candidates defaults missing tickers to enabled).
                     Pass None to use default.

    Returns:
        AllocationResult with:
        - rows: full output from size_selected (all candidates, both selected and rejected)
        - selected_count: number of rows with status == "SELECTED"
        - gross_exposure_pct: sum of alloc for status == "SELECTED" rows
        - rejection_counts: dict mapping rejection status -> count for all non-selected rows
    """
    if enabled_mask is None:
        enabled_mask = {}

    # Run the full pipeline: validate -> derive -> collapse -> size
    result_rows = select_candidates(candidates, config, enabled_mask)
    result_rows = size_selected(result_rows, config)

    # Compute summary statistics
    selected_rows = [row for row in result_rows if row.get("status") == "SELECTED"]
    selected_count = len(selected_rows)

    gross_exposure_pct = sum(row.get("alloc", 0.0) for row in selected_rows)

    # Rejection counts: all non-SELECTED rows
    rejection_counts = {}
    for row in result_rows:
        status = row.get("status")
        if status != "SELECTED":
            rejection_counts[status] = rejection_counts.get(status, 0) + 1

    return AllocationResult(
        rows=result_rows,
        selected_count=selected_count,
        gross_exposure_pct=gross_exposure_pct,
        rejection_counts=rejection_counts,
    )


def load_cluster_map(path: str) -> dict:
    """Load ticker-to-cluster mapping from a CSV file.

    Reads a two-column CSV file (no header) with format:
        ticker,cluster_name

    Missing tickers in the map should NOT crash size_selected(). The clustering
    logic in size_selected() implements a fallback: unmapped tickers form their
    own singleton cluster (using ticker name as cluster key).

    Args:
        path: file path to CSV file

    Returns:
        dict[str, str] mapping ticker -> cluster_name

    Raises:
        FileNotFoundError: if path does not exist
        ValueError: if CSV parsing fails (e.g., missing columns)
    """
    cluster_map = {}

    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row_num, row in enumerate(reader, start=1):
            if len(row) < 2:
                raise ValueError(f"Line {row_num} has fewer than 2 columns")
            ticker = row[0].strip()
            cluster_name = row[1].strip()
            if ticker:  # Skip empty ticker lines
                cluster_map[ticker] = cluster_name

    return cluster_map


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


# ===========================================================================
# E11-S08: Formula Template Engine (XLSX/ODS dialect rendering)
# ===========================================================================

"""Formula templates for portfolio allocation computations.

Each formula template is written once in a canonical XLSX form with:
- Cell references using Excel A1 notation (e.g., E20, F$3)
- Config cell references with absolute row anchors (e.g., $D$3, $D$4)
- Placeholder tokens for row substitution: {row}
- Function calls using comma argument separators (XLSX style)

Templates are rendered per dialect via render_formula(name, row, fmt):
- XLSX: Returns formula with '=' prefix and comma separators
- ODS: Returns formula with 'of:=' prefix and semicolon separators
  (per ODS spec; semicolons separate function arguments in many locales)

Column mapping (formula columns O through AJ, per E11-S08/S09/S10):
- O  = ev_net              EV net of costs (RFC §4.2)
- P  = kelly_raw           Binary Kelly before capping/multiplier
- Q  = score               Return per unit risk (ranking key)
- R  = alloc_pct           Final allocation % after pos/cluster/gross caps
- S  = alloc_eur           Allocation expressed as currency amount
- T  = flags               Composite flags (DATA_MISMATCH, POS_CAPPED, CLUSTER_CAPPED)
- U  = advised_liq         Carried upstream liquidity % (displayed but ignored)
- V  = risk_pct            ABS(stop - entry) / entry * 100
- W  = reward_pct          ABS(target - entry) / entry * 100
- X  = b                   Payoff ratio (geometry fallback in v1)
- Y  = loss_pct            Basis for score denominator
- Z  = shrink              Confidence weight n / (n + n0)
- AA = ev_shrunk           ev_pct * shrink
- AB = p_shrunk            Win rate shrunk toward 50%
- AC = kelly_frac          Fractional Kelly decimal (before cap)
- AD = alloc_raw_pct       Kelly_frac * 100
- AE = pos_capped_alloc    MIN(alloc_raw_pct, max_pos_pct)
- AF = pos_capped_flag     "POS_CAPPED" if capped
- AG = ev_implied          base_win_rate*reward - (1-base_win_rate)*risk
- AH = ev_ratio            ev_pct / ev_implied
- AI = data_mismatch       "DATA_MISMATCH" if ev_ratio outside [0.5, 2.0]
- AJ = cluster_scale       Cluster-level scale factor (SUMIFS over pos_capped_alloc)

Summary-block config cell layout (absolute references):
- $D$3: n0 (shrinkage constant)
- $D$4: round_trip_cost_pct (cost per round trip, %)
- $D$5: kelly_mult (fractional Kelly multiplier)
- $D$6: gross_cap_pct (total exposure cap, %)
- $D$7: max_pos_pct (per-position cap, %)
- $D$8: max_cluster_pct (per-cluster cap, %)
- $D$9: equity (optional account equity for currency column; blank = no amount)
- $D$14: gross_scale factor summary cell (proportional scale if over gross cap)

Data cell layout for row N (per RFC §5.2 static columns A-N):
- B: Cluster
- E: Entry price
- F: Stop loss price
- G: Target price
- K: n (number of backtest trades)
- L: Win raw (base_win_rate as a fraction, e.g., 0.47)
- N: EV raw % (ev_pct from backtest)
"""

# Template strings keyed by column letter (O..AJ) plus the summary factor.
# Each contains a single {row} placeholder substituted by render_formula().
_FORMULA_TEMPLATES = {
    # Visible Section A derived columns
    "O": "N{row}*Z{row}-$D$4",
    "P": "IFERROR(AB{row}-(1-AB{row})/X{row},0)",
    "Q": "IFERROR(O{row}/Y{row},0)",
    "R": "AE{row}*AJ{row}*$D$14",
    "S": 'IF($D$9="","",R{row}*$D$9/100)',
    "T": 'IF(AI{row}="","",AI{row})&IF(AF{row}="","",IF(AI{row}="",""," ")&AF{row})&IF(AJ{row}<1,IF(AND(AI{row}="",AF{row}=""),""," ")&"CLUSTER_CAPPED","")',
    "U": '""',

    # Helper columns (grouped/collapsed by the sheet writer)
    "V": "IF(E{row}=0,0,ABS(F{row}-E{row})/E{row}*100)",
    "W": "IF(E{row}=0,0,ABS(G{row}-E{row})/E{row}*100)",
    "X": "IF(Y{row}=0,0,W{row}/Y{row})",
    "Y": "IF(E{row}=0,0,V{row})",
    "Z": "IF(K{row}=0,0,K{row}/(K{row}+$D$3))",
    "AA": "N{row}*Z{row}",
    "AB": "0.5+(L{row}-0.5)*Z{row}",
    "AC": "MAX(P{row},0)*$D$5",
    "AD": "AC{row}*100",
    "AE": "MIN(AD{row},$D$7)",
    "AF": 'IF(AD{row}>$D$7,"POS_CAPPED","")',
    "AG": "L{row}*W{row}-(1-L{row})*V{row}",
    "AH": "IFERROR(N{row}/AG{row},0)",
    "AI": 'IF(OR(AH{row}<0.5,AH{row}>2),"DATA_MISMATCH","")',
    "AJ": "IF(SUMIFS(AE$20:AE$400,B$20:B$400,B{row})>$D$8,$D$8/SUMIFS(AE$20:AE$400,B$20:B$400,B{row}),1)",

    # Summary-block gross scale factor (rendered into $D$14).
    # Must scale the post-cluster-cap total (AE * AJ) down to gross_cap_pct.
    # SUM(AJ) alone is wrong: it is just the count of cluster-scale factors.
    "gross_scale": "IF(SUMPRODUCT(AE$20:AE$400,AJ$20:AJ$400)>$D$6,$D$6/SUMPRODUCT(AE$20:AE$400,AJ$20:AJ$400),1)",
}


# Concept-name aliases for callers that prefer readable names.  Each alias
# resolves to one of the canonical column-letter keys above, so the same
# single template is used for both naming styles.
_FORMULA_ALIASES = {
    "ev_net": "O",
    "kelly_raw": "P",
    "score": "Q",
    "alloc_pct": "R",
    "alloc_eur": "S",
    "flags": "T",
    "advised_liq_pct": "U",
    "risk_pct": "V",
    "reward_pct": "W",
    "b": "X",
    "loss_pct": "Y",
    "shrink": "Z",
    "ev_shrunk": "AA",
    "p_shrunk": "AB",
    "kelly_frac": "AC",
    "alloc_raw_pct": "AD",
    "pos_capped_alloc": "AE",
    "pos_capped": "AF",
    "ev_implied": "AG",
    "ev_ratio": "AH",
    "data_mismatch": "AI",
    "cluster_scale": "AJ",
    "gross_scale": "gross_scale",
}


def render_formula(name: str, row: int, fmt: str) -> str:
    """Render a formula template for a given row number and dialect.

    Implements E11-S08 acceptance criteria: both XLSX and ODS dialects
    derive from one shared template, with only dialect-specific syntax changes.

    Args:
        name: Formula name.  Accepts canonical column-letter keys ("O".."AJ",
            "gross_scale") and concept aliases ("risk_pct", "ev_net", ...).
        row: Data row number (20..400 per RFC §5.1), used for cell reference substitution.
        fmt: Output dialect, "xlsx" or "ods".

    Returns:
        Formula string ready for insertion into a spreadsheet cell:
        - XLSX: "=<formula>" with comma-separated function arguments
        - ODS: "of:=<formula>" with semicolon-separated function arguments

    Raises:
        ValueError: if name cannot be resolved or fmt not in ("xlsx", "ods")

    Notes:
        - Row-number substitution is exact: {row} placeholders become the literal row number
        - Config cell references like $D$3 are preserved as-is
        - Both dialects use the same underlying template; no separate formula sets
        - Compliance: forbids FILTER, SORT, UNIQUE, LET, LAMBDA, XLOOKUP, MAXIFS, MINIFS
    """
    if fmt not in ("xlsx", "ods"):
        raise ValueError(f"fmt must be 'xlsx' or 'ods', got {fmt!r}")

    canonical = _FORMULA_ALIASES.get(name, name)
    if canonical not in _FORMULA_TEMPLATES:
        raise ValueError(f"formula name {name!r} not found in templates")

    template = _FORMULA_TEMPLATES[canonical]
    formula = template.format(row=row)

    if fmt == "xlsx":
        return "=" + formula
    return "of:=" + _convert_commas_to_semicolons(formula)


def _convert_commas_to_semicolons(formula: str) -> str:
    """Convert comma argument separators to semicolons for ODS format.

    The canonical templates use commas only as function-argument separators and
    inside decimal numbers (e.g. 0.5) use a dot, so a global replacement is safe.
    No string literal contains a comma.
    """
    return formula.replace(",", ";")


def get_formula_names() -> list[str]:
    """Return the sorted list of canonical formula names (column letters + gross_scale)."""
    return sorted(_FORMULA_TEMPLATES.keys())


def get_formula_aliases() -> dict[str, str]:
    """Return the concept-name -> canonical-key alias map."""
    return dict(_FORMULA_ALIASES)


# =============================================================================
# E11-S09 / E11-S10: Shared Sheet Layout Constants
# =============================================================================

# Config block layout.  Order is chosen so that the first seven editable value
# cells ($D$3..$D$9) line up with the absolute references used by the formula
# templates above.
_CONFIG_BLOCK = [
    ("n0", "n0"),
    ("round_trip_cost_pct", "round_trip_cost_pct"),
    ("kelly_mult", "kelly_mult"),
    ("gross_cap_pct", "gross_cap_pct"),
    ("max_pos_pct", "max_pos_pct"),
    ("max_cluster_pct", "max_cluster_pct"),
    ("equity", "equity"),
    ("min_n", "min_n"),
    ("top_k", "top_k"),
    ("dust_min_pct", "dust_min_pct"),
    ("cluster_map", "cluster_map"),
]

_DATA_START_ROW = 19
_HEADER_ROW = 19

# Static columns A-N and their human-readable header.
_STATIC_HEADERS = [
    "Ticker", "Cluster", "Strategy", "Dir", "Entry", "Stop", "Target",
    "Risk %", "Reward %", "b", "n", "Win raw", "Win shrunk", "EV raw %",
]

# Headers for the formula-driven columns O-U (visible Section A derived columns).
_FORMULA_HEADERS = {
    "O": "EV net %",
    "P": "Kelly raw",
    "Q": "Score",
    "R": "Alloc %",
    "S": "Alloc EUR",
    "T": "Flags",
    "U": "Advised liq % (ignored)",
}

# Headers for the helper formula columns V-AJ.
_HELPER_HEADERS = [
    "risk_pct", "reward_pct", "b", "loss_pct", "shrink", "ev_shrunk",
    "p_shrunk", "kelly_frac", "alloc_raw_pct", "pos_capped_alloc",
    "pos_capped_flag", "ev_implied", "ev_ratio", "data_mismatch", "cluster_scale",
]

_FORMULA_COLS = [
    "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "AA", "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ",
]

# Summary-block formulas shared by XLSX and ODS writers.  Dialect-specific
# rendering (comma vs. semicolon separators, '=' vs. 'of:=' prefix) is applied
# by the writers so the formula logic is not duplicated.
_SUMMARY_FORMULAS = {
    "selected_count": '=COUNTIF(R20:R400,">0")',
    "gross_exposure": "=SUM(R20:R400)",
    "enabled_count": "=COUNTA(N20:N400)",
}


def _xlsx_column_letter_to_index(col_letter: str) -> int:
    """Convert an Excel column letter (A, AA, AJ) to a 1-based column index."""
    idx = 0
    for ch in col_letter:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx


def _empty_if_none(value):
    """Return an empty string for None, otherwise the value unchanged."""
    if value is None:
        return ""
    return value


def _xlsx_formula_to_ods(formula: str) -> str:
    """Convert an XLSX-style formula string into an ODS formula string.

    Applies the same comma-to-semicolon conversion used by render_formula()
    for ODS output and prefixes the result with ``of:=``.
    """
    if not formula.startswith("="):
        return formula
    return "of:=" + _convert_commas_to_semicolons(formula[1:])


# =============================================================================
# E11-S09: XLSX Sheet Writer
# =============================================================================


def write_xlsx_sheet(workbook, result: AllocationResult, config: AllocationConfig,
                     report_date, generator_version: str):
    """Write the ``Allocation`` sheet into an existing ``openpyxl.Workbook``.

    Layout follows RFC allocation_sheet.md §5 and ticket E11-S09:
      - Row 1: title, report date, generator version.
      - Rows 3-13: config block (parameter name, editable value, locked default).
      - Rows 14-16: summary formulas (selected count, gross exposure %, enabled
        count, and the gross scale factor in $D$14).
      - Row 18: instruction line.
      - Row 19: header row A-AJ.
      - Rows 20..N: one row per candidate in ``result.rows`` in the order given.
      - Below the data: Section B cluster-exposure table.
      - Helper columns X-AF are grouped (outlined) but not hidden.
      - All cells are locked except the editable config value cells (D3:D13) and
        column N (the per-row editable input column).

    The function performs no disk I/O; it mutates the provided in-memory
    ``Workbook``.
    """
    from openpyxl import Workbook

    if not isinstance(workbook, Workbook):
        raise TypeError("workbook must be an openpyxl Workbook")

    ws = workbook.create_sheet("Allocation")

    # -------------------------------------------------------------------------
    # Row 1: title line
    # -------------------------------------------------------------------------
    ws["A1"] = "Portfolio Allocation"
    ws["B1"] = _empty_if_none(report_date)
    ws["C1"] = f"generator {generator_version}"

    # -------------------------------------------------------------------------
    # Rows 3-13: config block
    # -------------------------------------------------------------------------
    # Parameter names in column C, editable values in column D, shipped defaults
    # (locked) in column E.
    default_config = AllocationConfig()
    for offset, (label, attr) in enumerate(_CONFIG_BLOCK):
        row = 3 + offset
        ws.cell(row=row, column=3, value=label)
        value = getattr(config, attr)
        if attr == "cluster_map":
            value = str(value) if value else ""
        ws.cell(row=row, column=4, value=_empty_if_none(value))
        default_value = getattr(default_config, attr)
        if attr == "cluster_map":
            default_value = str(default_value) if default_value else ""
        ws.cell(row=row, column=5, value=_empty_if_none(default_value))

    # -------------------------------------------------------------------------
    # Rows 14-16: summary block
    # -------------------------------------------------------------------------
    ws["A14"] = "Selected count"
    ws["C14"] = _SUMMARY_FORMULAS["selected_count"]
    ws["A15"] = "Gross exposure %"
    ws["C15"] = _SUMMARY_FORMULAS["gross_exposure"]
    ws["A16"] = "Enabled count"
    ws["C16"] = _SUMMARY_FORMULAS["enabled_count"]
    ws["D14"] = render_formula("gross_scale", 14, "xlsx")

    # -------------------------------------------------------------------------
    # Row 18: instruction line
    # -------------------------------------------------------------------------
    ws["A18"] = (
        "Edit only the config values (column D) and the per-row input column "
        "(column N). All other cells are computed."
    )

    # -------------------------------------------------------------------------
    # Row 19: header row
    # -------------------------------------------------------------------------
    headers = (
        _STATIC_HEADERS
        + [_FORMULA_HEADERS[col] for col in _FORMULA_COLS[:7]]
        + _HELPER_HEADERS
    )
    for col_idx, header in enumerate(headers, start=1):
        ws.cell(row=_HEADER_ROW, column=col_idx, value=header)

    # -------------------------------------------------------------------------
    # Rows 20..N: candidate rows
    # -------------------------------------------------------------------------
    def _cluster_for_ticker(ticker: str) -> str:
        return config.cluster_map.get(ticker, ticker)

    for row_offset, row_data in enumerate(result.rows):
        excel_row = _DATA_START_ROW + 1 + row_offset
        derived = row_data.get("derived", {}) or {}
        ticker = row_data.get("ticker", "")
        direction = row_data.get("direction", "")

        static_values = [
            ticker,
            _cluster_for_ticker(ticker),
            row_data.get("strategy", ""),
            direction.capitalize() if isinstance(direction, str) else "",
            row_data.get("entry"),
            row_data.get("stop"),
            row_data.get("target"),
            derived.get("risk_pct"),
            derived.get("reward_pct"),
            derived.get("b"),
            row_data.get("n"),
            row_data.get("base_win_rate"),
            derived.get("p_shrunk"),
            row_data.get("ev_pct"),
        ]
        for col_idx, value in enumerate(static_values, start=1):
            ws.cell(row=excel_row, column=col_idx, value=_empty_if_none(value))

        for col_letter in _FORMULA_COLS:
            col_idx = _xlsx_column_letter_to_index(col_letter)
            ws.cell(row=excel_row, column=col_idx,
                    value=render_formula(col_letter, excel_row, "xlsx"))

    # -------------------------------------------------------------------------
    # Section B: cluster exposure table
    # -------------------------------------------------------------------------
    data_end_row = _DATA_START_ROW + len(result.rows)
    cluster_header_row = data_end_row + 2
    ws.cell(row=cluster_header_row, column=1, value="Cluster")
    ws.cell(row=cluster_header_row, column=2, value="Positions")
    ws.cell(row=cluster_header_row, column=3, value="Gross %")
    ws.cell(row=cluster_header_row, column=4, value="Cap %")
    ws.cell(row=cluster_header_row, column=5, value="Capped?")

    selected_rows = [r for r in result.rows if r.get("status") == "SELECTED"]
    clusters = sorted(set(config.cluster_map.values())) if config.cluster_map else []
    for cluster_offset, cluster in enumerate(clusters):
        row = cluster_header_row + 1 + cluster_offset
        cluster_rows = [
            r for r in selected_rows
            if _cluster_for_ticker(r.get("ticker", "")) == cluster
        ]
        positions = len(cluster_rows)
        gross = sum(r.get("alloc", 0.0) for r in cluster_rows)
        capped = any("CLUSTER_CAPPED" in r.get("flags", []) for r in cluster_rows)
        ws.cell(row=row, column=1, value=cluster)
        ws.cell(row=row, column=2, value=positions)
        ws.cell(row=row, column=3, value=gross)
        ws.cell(row=row, column=4, value=config.max_cluster_pct)
        ws.cell(row=row, column=5, value="yes" if capped else "no")

    # -------------------------------------------------------------------------
    # Section C: rejected signals (compact audit trail)
    # -------------------------------------------------------------------------
    if cluster_header_row + len(clusters) + 1 >= ws.max_row:
        rejected_header_row = cluster_header_row + len(clusters) + 2
    else:
        rejected_header_row = ws.max_row + 2

    ws.cell(row=rejected_header_row, column=1, value="Ticker")
    ws.cell(row=rejected_header_row, column=2, value="Strategy")
    ws.cell(row=rejected_header_row, column=3, value="Dir")
    ws.cell(row=rejected_header_row, column=4, value="Score")
    ws.cell(row=rejected_header_row, column=5, value="Reason")

    rejected = [r for r in result.rows if r.get("status") != "SELECTED"]
    rejected.sort(
        key=lambda r: (
            r.get("status", ""),
            -(r.get("derived", {}) or {}).get("score", float("-inf")),
        )
    )
    for rej_offset, r in enumerate(rejected):
        row = rejected_header_row + 1 + rej_offset
        direction = r.get("direction", "")
        score = (r.get("derived", {}) or {}).get("score")
        ws.cell(row=row, column=1, value=r.get("ticker", ""))
        ws.cell(row=row, column=2, value=r.get("strategy", ""))
        ws.cell(row=row, column=3,
                value=direction.capitalize() if isinstance(direction, str) else "")
        ws.cell(row=row, column=4, value=_empty_if_none(score))
        ws.cell(row=row, column=5, value=r.get("status", ""))

    # -------------------------------------------------------------------------
    # Column grouping: helper columns V-AF are outlined but not hidden.
    # -------------------------------------------------------------------------
    ws.column_dimensions.group("V", "AF", outline_level=1, hidden=False)

    # -------------------------------------------------------------------------
    # Sheet protection: lock everything except the editable config values (D3:D13)
    # and column N.
    # -------------------------------------------------------------------------
    for cell in ws["N"]:
        cell.protection = Protection(locked=False)

    for config_row in range(3, 3 + len(_CONFIG_BLOCK)):
        ws.cell(row=config_row, column=4).protection = Protection(locked=False)

    ws.protection.sheet = True


# =============================================================================
# E11-S10: ODS Sheet Writer
# =============================================================================

def _ods_create_cell(value, formula=None):
    """Create an odfpy ``TableCell`` with optional ODS formula."""
    from odf.table import TableCell
    from odf.text import P

    text = "" if value is None else str(value)

    if formula:
        # Formula cells carry a cached float value of 0.0 and the formula
        # attribute; the displayed text is left empty.
        cell = TableCell(valuetype="float", value=0.0)
        cell.setAttribute("formula", formula)
        cell.addElement(P(text=text))
    elif isinstance(value, bool):
        cell = TableCell(valuetype="boolean", value="true" if value else "false")
        cell.addElement(P(text=text))
    elif isinstance(value, (int, float)):
        cell = TableCell(valuetype="float", value=float(value))
        cell.addElement(P(text=text))
    else:
        cell = TableCell(valuetype="string")
        cell.addElement(P(text=text))
    return cell


def _ods_add_row(table, values, start_col=0, formulas=None):
    """Append a row to an ODS ``Table`` with leading empty cells and formulas.

    Args:
        table: odfpy ``Table`` instance.
        values: iterable of cell values placed starting at ``start_col``.
        start_col: 0-based column index where ``values`` begin.
        formulas: dict mapping absolute 0-based column index to an ODS formula
            string (``of:=...``).  The matching cell is created as a formula
            cell with the corresponding ``values`` entry as display text.
    """
    from odf.table import TableRow

    formulas = formulas or {}
    row = TableRow()
    end_col = max(len(values) + start_col, max(formulas, default=-1) + 1)
    for col_idx in range(end_col):
        if start_col <= col_idx < start_col + len(values):
            value = values[col_idx - start_col]
        else:
            value = ""
        cell = _ods_create_cell(value, formula=formulas.get(col_idx))
        row.addElement(cell)
    table.addElement(row)
    return row


def write_ods_sheet(document, result: AllocationResult, config: AllocationConfig,
                    report_date, generator_version: str):
    """Write the ``Allocation`` sheet into an existing ``odfpy`` document.

    Layout mirrors ``write_xlsx_sheet`` exactly (RFC allocation_sheet.md §5 and
    ticket E11-S10):
      - Row 1: title, report date, generator version.
      - Row 2: visible note that the sheet is unprotected.
      - Rows 3-13: config block (parameter name, editable value, shipped default).
      - Rows 14-16: summary formulas (selected count, gross exposure %, enabled
        count, and the gross scale factor in $D$14).
      - Row 18: instruction line.
      - Row 19: header row A-AJ.
      - Rows 20..N: one row per candidate in ``result.rows`` in the order given.
      - Below the data: Section B cluster-exposure table and Section C rejected
        signals table.

    No sheet protection is applied because odfpy protection support is
    unreliable (RFC §4.6); the unprotected state is surfaced to the user via
    the row-2 note.

    The function performs no disk I/O; it mutates the provided in-memory
    ``OpenDocumentSpreadsheet``.
    """
    from odf import opendocument
    from odf.table import Table

    if not isinstance(document, opendocument.OpenDocument):
        raise TypeError("document must be an odfpy OpenDocumentSpreadsheet")
    if document.mimetype != "application/vnd.oasis.opendocument.spreadsheet":
        raise TypeError("document must be an odfpy OpenDocumentSpreadsheet")

    table = Table(name="Allocation")
    document.spreadsheet.addElement(table)

    # -------------------------------------------------------------------------
    # Row 1: title line
    # -------------------------------------------------------------------------
    _ods_add_row(
        table,
        ["Portfolio Allocation", _empty_if_none(report_date), f"generator {generator_version}"],
    )

    # -------------------------------------------------------------------------
    # Row 2: unprotected-sheet note (ODS protection is unreliable)
    # -------------------------------------------------------------------------
    _ods_add_row(
        table,
        ["Note: this sheet is intentionally unprotected (ODS protection support is unreliable)."],
    )

    # -------------------------------------------------------------------------
    # Rows 3-13: config block
    # -------------------------------------------------------------------------
    # Parameter names in column C, editable values in column D, shipped defaults
    # in column E.
    default_config = AllocationConfig()
    for label, attr in _CONFIG_BLOCK:
        value = getattr(config, attr)
        if attr == "cluster_map":
            value = str(value) if value else ""
        default_value = getattr(default_config, attr)
        if attr == "cluster_map":
            default_value = str(default_value) if default_value else ""
        _ods_add_row(
            table,
            [label, _empty_if_none(value), _empty_if_none(default_value)],
            start_col=2,
        )

    # -------------------------------------------------------------------------
    # Rows 14-16: summary block
    # -------------------------------------------------------------------------
    _ods_add_row(
        table,
        ["Selected count", ""],
        formulas={
            2: _xlsx_formula_to_ods(_SUMMARY_FORMULAS["selected_count"]),
            3: render_formula("gross_scale", 14, "ods"),
        },
    )
    _ods_add_row(
        table,
        ["Gross exposure %", ""],
        formulas={2: _xlsx_formula_to_ods(_SUMMARY_FORMULAS["gross_exposure"])},
    )
    _ods_add_row(
        table,
        ["Enabled count", ""],
        formulas={2: _xlsx_formula_to_ods(_SUMMARY_FORMULAS["enabled_count"])},
    )

    # -------------------------------------------------------------------------
    # Row 17: blank separator (mirrors XLSX layout)
    # -------------------------------------------------------------------------
    _ods_add_row(table, [""])

    # -------------------------------------------------------------------------
    # Row 18: instruction line
    # -------------------------------------------------------------------------
    _ods_add_row(
        table,
        [
            "Edit only the config values (column D) and the per-row input column "
            "(column N). All other cells are computed."
        ],
    )

    # -------------------------------------------------------------------------
    # Row 19: header row
    # -------------------------------------------------------------------------
    headers = (
        _STATIC_HEADERS
        + [_FORMULA_HEADERS[col] for col in _FORMULA_COLS[:7]]
        + _HELPER_HEADERS
    )
    _ods_add_row(table, headers)

    # -------------------------------------------------------------------------
    # Rows 20..N: candidate rows
    # -------------------------------------------------------------------------
    def _cluster_for_ticker(ticker: str) -> str:
        return config.cluster_map.get(ticker, ticker)

    for row_offset, row_data in enumerate(result.rows):
        excel_row = _DATA_START_ROW + 1 + row_offset
        derived = row_data.get("derived", {}) or {}
        ticker = row_data.get("ticker", "")
        direction = row_data.get("direction", "")

        static_values = [
            ticker,
            _cluster_for_ticker(ticker),
            row_data.get("strategy", ""),
            direction.capitalize() if isinstance(direction, str) else "",
            row_data.get("entry"),
            row_data.get("stop"),
            row_data.get("target"),
            derived.get("risk_pct"),
            derived.get("reward_pct"),
            derived.get("b"),
            row_data.get("n"),
            row_data.get("base_win_rate"),
            derived.get("p_shrunk"),
            row_data.get("ev_pct"),
        ]

        # Visible + helper formula columns O..AJ are formula cells.
        formulas = {}
        for col_letter in _FORMULA_COLS:
            col_idx = _xlsx_column_letter_to_index(col_letter) - 1
            formulas[col_idx] = render_formula(col_letter, excel_row, "ods")

        values = static_values + [""] * len(_FORMULA_COLS)
        _ods_add_row(table, values, formulas=formulas)

    # -------------------------------------------------------------------------
    # Section B: cluster exposure table
    # -------------------------------------------------------------------------
    _ods_add_row(table, [""])  # blank separator row
    _ods_add_row(
        table,
        ["Cluster", "Positions", "Gross %", "Cap %", "Capped?"],
    )

    selected_rows = [r for r in result.rows if r.get("status") == "SELECTED"]
    clusters = sorted(set(config.cluster_map.values())) if config.cluster_map else []
    for cluster in clusters:
        cluster_rows = [
            r for r in selected_rows
            if _cluster_for_ticker(r.get("ticker", "")) == cluster
        ]
        positions = len(cluster_rows)
        gross = sum(r.get("alloc", 0.0) for r in cluster_rows)
        capped = any("CLUSTER_CAPPED" in r.get("flags", []) for r in cluster_rows)
        _ods_add_row(
            table,
            [cluster, positions, gross, config.max_cluster_pct, "yes" if capped else "no"],
        )

    # -------------------------------------------------------------------------
    # Section C: rejected signals (compact audit trail)
    # -------------------------------------------------------------------------
    _ods_add_row(table, ["Ticker", "Strategy", "Dir", "Score", "Reason"])

    rejected = [r for r in result.rows if r.get("status") != "SELECTED"]
    rejected.sort(
        key=lambda r: (
            r.get("status", ""),
            -(r.get("derived", {}) or {}).get("score", float("-inf")),
        )
    )
    for r in rejected:
        direction = r.get("direction", "")
        score = (r.get("derived", {}) or {}).get("score")
        _ods_add_row(
            table,
            [
                r.get("ticker", ""),
                r.get("strategy", ""),
                direction.capitalize() if isinstance(direction, str) else "",
                _empty_if_none(score),
                r.get("status", ""),
            ],
        )


# =============================================================================
# E11-S11: Markdown Section Writer
# =============================================================================

def write_md_section(result: AllocationResult, config: AllocationConfig) -> str:
    """Render the RFC §6 Markdown "Portfolio Allocation" section.

    Returns a static snapshot of ``result`` as a markdown string, including the
    config summary, selection summary, selected-position table, cluster
    exposure line, and compact rejection-count summary.

    Args:
        result: AllocationResult from ``allocate()``.
        config: AllocationConfig used to produce ``result``.

    Returns:
        Markdown string ready to append to a signals report.
    """
    # Heading
    lines = ["## Portfolio Allocation", ""]

    # Config summary line (RFC §6 example format)
    lines.append(
        f"Config: n0={config.n0:g} min_n={config.min_n:g} "
        f"cost={config.round_trip_cost_pct:g}% kelly_mult={config.kelly_mult:g} "
        f"top_k={config.top_k:g} max_pos={config.max_pos_pct:g}% "
        f"max_cluster={config.max_cluster_pct:g}% gross_cap={config.gross_cap_pct:g}%"
    )
    lines.append("")

    # Selection summary
    total_signals = len(result.rows)
    lines.append(
        f"Selected {result.selected_count} of {total_signals} signals. "
        f"Gross exposure: {result.gross_exposure_pct:.1f}%."
    )
    lines.append("")

    # Selected-position table via kairos_signals.format_table
    headers = ["Ticker", "Dir", "Strategy", "Entry", "Stop", "Target", "EV net", "Score", "Alloc"]
    align = ["l", "l", "l", "r", "r", "r", "r", "r", "r"]

    selected_rows = [row for row in result.rows if row.get("status") == "SELECTED"]
    table_rows = []
    for row in selected_rows:
        direction = row.get("direction", "")
        derived = row.get("derived", {}) or {}
        ev_net = derived.get("ev_net")
        score = derived.get("score")
        alloc = row.get("alloc", 0.0)

        entry = row.get("entry")
        stop = row.get("stop")
        target = row.get("target")

        table_rows.append({
            "Ticker": row.get("ticker", ""),
            "Dir": direction.capitalize() if isinstance(direction, str) else "",
            "Strategy": row.get("strategy", ""),
            "Entry": f"{entry:.2f}" if entry is not None else "",
            "Stop": f"{stop:.2f}" if stop is not None else "",
            "Target": f"{target:.2f}" if target is not None else "",
            "EV net": f"{ev_net:.2f}%" if ev_net is not None else "",
            "Score": f"{score:.2f}" if score is not None else "",
            "Alloc": f"{alloc:.1f}%",
        })

    table_lines = format_table(headers, table_rows, align)
    lines.extend(table_lines)
    lines.append("")

    # Cluster exposure line: sum alloc per cluster among selected rows
    cluster_sums: dict[str, float] = {}
    for row in selected_rows:
        ticker = row.get("ticker", "")
        cluster = config.cluster_map.get(ticker, ticker)
        cluster_sums[cluster] = cluster_sums.get(cluster, 0.0) + row.get("alloc", 0.0)

    if cluster_sums:
        sorted_clusters = sorted(cluster_sums.items(), key=lambda kv: (-kv[1], kv[0]))
        cluster_parts = [f"{cluster} {total:.1f}%" for cluster, total in sorted_clusters]
        lines.append(f"Cluster exposure: {', '.join(cluster_parts)}")
    else:
        lines.append("Cluster exposure: none")
    lines.append("")

    # Rejection-count summary line
    non_selected_count = total_signals - result.selected_count
    if result.rejection_counts:
        sorted_reasons = sorted(
            result.rejection_counts.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        reason_parts = [f"{reason} {count}" for reason, count in sorted_reasons]
        lines.append(f"Rejected: {non_selected_count} total -- {', '.join(reason_parts)}")
    else:
        lines.append(f"Rejected: {non_selected_count} total --")

    return "\n".join(lines)
