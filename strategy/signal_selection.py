"""signal_selection.py — Configurable signal-selection DSL for allocation.py.

Parses a small rule string like:

    "'n' > 60, 'Win raw' > 0.6, ORDER 'EV raw %' DESC, TOP 3"

into a SignalSelectionRule that strategy/allocation.py's select_candidates()
can use to replace the hardcoded min_n/ev_net gate, the score-based rank key,
and the top_k cutoff.

Grammar:
    selection   := group (';' group)* (',' order_clause)? (',' top_clause)?
    group       := condition (',' condition)*
    condition   := "'col'" OP value        # quotes may be ' or "
    OP          := '>' | '>=' | '<' | '<=' | '==' | '!='
    order_clause:= 'ORDER' "'col'" ('ASC' | 'DESC')?     # default DESC, case-insensitive
    top_clause  := 'TOP' <int>                            # case-insensitive

Conditions within one semicolon-separated group are AND'd; groups themselves
are OR'd (a candidate matches the rule if it satisfies ALL conditions in ANY
one group). ORDER/TOP apply to the whole rule regardless of which group(s)
they appear in -- there is only ever one ORDER and one TOP for the entire
spec, found by scanning every group (added 2026-08-22 for an exact-whitelist
use case: matching one of N explicit (Strategy, Ticker, Model) triples,
which needs OR-of-AND-groups since the grammar's conditions alone are
AND-only and per-column, with no cross-column tuple matching).

Column names are matched case-insensitively against a fixed registry that
mirrors the RFC allocation_sheet.md §5.2 "Allocation" sheet columns available
before sizing (Ticker..Sharpe), plus `Model` (added 2026-08-22, mirrors
allocation.Candidate.model -- "Base" or "Finetuned(<assets>)"; not part of
the original RFC schema, but needed to distinguish a base vs finetuned
signal for the same strategy+ticker, which nothing else in the registry
could express). Kept separate from allocation.py so the grammar/registry are
unit-testable in isolation.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Union


class SignalSelectionError(ValueError):
    """Raised for any malformed or semantically invalid --signal-selection spec."""


@dataclass
class SelectionCondition:
    column: str  # canonical registry key, e.g. "n", "Win raw"
    op: str  # one of '>','>=','<','<=','==','!='
    value: Union[float, str]


@dataclass
class SignalSelectionRule:
    conditions: list  # the first (or only) group's AND'd conditions -- kept for
        # backward compat with the pre-OR single-group grammar; always equal to
        # condition_groups[0]. Prefer condition_groups (or rule_matches()) for
        # new code, since this field alone can't represent OR semantics.
    order_by: Optional[tuple]  # (column, descending); None if no ORDER clause
    top_n: Optional[int]  # None if no TOP clause
    raw: str = ""  # original spec string, for report echoing
    # list[list[SelectionCondition]]. Conditions within one inner list are
    # AND'd; the outer list's groups are OR'd. Always populated (a
    # single-group spec, the common case, has exactly one inner list,
    # identical to `conditions`).
    condition_groups: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Column registry
# ---------------------------------------------------------------------------
# display name -> (kind, resolver(candidate, derived, cluster_map) -> value)
_TEXT_COLUMNS = {
    "Ticker": lambda c, derived, cluster_map: c.ticker,
    "Cluster": lambda c, derived, cluster_map: cluster_map.get(c.ticker, ""),
    "Strategy": lambda c, derived, cluster_map: c.strategy,
    "Dir": lambda c, derived, cluster_map: c.direction,
    "Model": lambda c, derived, cluster_map: c.model or "",
}

_NUMERIC_COLUMNS = {
    "Entry": lambda c, derived, cluster_map: c.entry,
    "Stop": lambda c, derived, cluster_map: c.stop,
    "Target": lambda c, derived, cluster_map: c.target,
    "Risk %": lambda c, derived, cluster_map: derived["risk_pct"],
    "Reward %": lambda c, derived, cluster_map: derived["reward_pct"],
    "b": lambda c, derived, cluster_map: derived["b"],
    "n": lambda c, derived, cluster_map: c.n,
    "Win raw": lambda c, derived, cluster_map: c.base_win_rate,
    "Win shrunk": lambda c, derived, cluster_map: derived["p_shrunk"],
    "EV raw %": lambda c, derived, cluster_map: c.ev_pct,
    "EV net %": lambda c, derived, cluster_map: derived["ev_net"],
    "Kelly raw": lambda c, derived, cluster_map: derived["kelly_raw"],
    "Score": lambda c, derived, cluster_map: derived["score"],
    "Sharpe": lambda c, derived, cluster_map: c.sharpe,
}

_REGISTRY = {}
for _name in _TEXT_COLUMNS:
    _REGISTRY[_name.lower()] = ("text", _name, _TEXT_COLUMNS[_name])
for _name in _NUMERIC_COLUMNS:
    _REGISTRY[_name.lower()] = ("numeric", _name, _NUMERIC_COLUMNS[_name])

_VALID_COLUMN_NAMES = sorted(list(_TEXT_COLUMNS.keys()) + list(_NUMERIC_COLUMNS.keys()))

_VALID_OPS = {">", ">=", "<", "<=", "==", "!="}
_TEXT_ONLY_OPS = {"==", "!="}  # ops valid on text columns


def column_kind(column: str) -> str:
    """Return 'text' or 'numeric' for a canonical registry column name.

    Raises SignalSelectionError if the column is not in the registry.
    """
    entry = _REGISTRY.get(column.lower())
    if entry is None:
        raise SignalSelectionError(
            f"Unknown column {column!r} in --signal-selection. "
            f"Valid columns: {', '.join(_VALID_COLUMN_NAMES)}"
        )
    return entry[0]


def resolve_column(column: str, candidate, derived: dict, cluster_map: dict):
    """Resolve a canonical column name to its value for one candidate row.

    Raises SignalSelectionError if the column is not in the registry.
    """
    entry = _REGISTRY.get(column.lower())
    if entry is None:
        raise SignalSelectionError(
            f"Unknown column {column!r} in --signal-selection. "
            f"Valid columns: {', '.join(_VALID_COLUMN_NAMES)}"
        )
    _, canonical_name, resolver = entry
    return resolver(candidate, derived, cluster_map)


def _canonical_column_name(column: str) -> str:
    entry = _REGISTRY.get(column.lower())
    if entry is None:
        raise SignalSelectionError(
            f"Unknown column {column!r} in --signal-selection. "
            f"Valid columns: {', '.join(_VALID_COLUMN_NAMES)}"
        )
    return entry[1]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
_CONDITION_RE = re.compile(
    r"""^\s*['"](?P<col>[^'"]+)['"]\s*(?P<op>>=|<=|==|!=|>|<)\s*(?P<val>.+)$"""
)
_ORDER_RE = re.compile(r"^\s*ORDER\b", re.IGNORECASE)
_TOP_RE = re.compile(r"^\s*TOP\b", re.IGNORECASE)

_ORDER_FULL_RE = re.compile(
    r"""^\s*ORDER\s+['"](?P<col>[^'"]+)['"]\s*(?P<dir>ASC|DESC)?\s*$""",
    re.IGNORECASE,
)
_TOP_FULL_RE = re.compile(r"^\s*TOP\s+(?P<n>.+?)\s*$", re.IGNORECASE)


def _split_top_level_commas(spec: str) -> list:
    """Split spec on top-level commas.

    All commas in this grammar are top-level (no nested parens/brackets), so
    this is a plain split — kept as a helper for clarity and in case the
    grammar grows nesting later.
    """
    return [part.strip() for part in spec.split(",")]


def _split_top_level_semicolons(spec: str) -> list:
    """Split spec on top-level semicolons, separating OR'd condition groups.

    Plain split, same reasoning as _split_top_level_commas: no value in this
    grammar is expected to contain a literal semicolon. Note this shares that
    function's existing, pre-dating limitation: a quoted value containing a
    comma (e.g. a multi-symbol finetuned group's Model label,
    'Finetuned(BTC-USD,ETH-USD)') will be incorrectly split mid-value by the
    comma-splitter that runs on each group afterward -- not something this
    change introduces or fixes, just inherited from the existing grammar,
    which has never supported quoted commas. Single-symbol Model values (the
    only kind exercised so far) are unaffected.
    """
    return [part.strip() for part in spec.split(";")]


def _parse_value(raw_val: str):
    """Parse a condition's RHS value: float if possible, else a string literal."""
    raw_val = raw_val.strip()
    try:
        return float(raw_val)
    except ValueError:
        pass
    if len(raw_val) >= 2 and raw_val[0] == raw_val[-1] and raw_val[0] in ("'", '"'):
        return raw_val[1:-1]
    return raw_val


def parse_signal_selection(spec: str) -> SignalSelectionRule:
    """Parse a --signal-selection spec string into a SignalSelectionRule.

    Raises SignalSelectionError with a clear, specific message for any
    malformed or semantically invalid clause.

    Semicolons split the spec into OR'd groups (each internally comma-split
    into AND'd conditions, as before); ORDER/TOP are scanned across every
    group and apply to the whole rule regardless of which group they appear
    in -- there is still only ever one ORDER and one TOP total. A single-
    group spec (no semicolons, the common case and the only form this
    grammar supported before 2026-08-22) behaves identically to before,
    including a 0-condition rule (just ORDER/TOP) meaning "no filtering."
    """
    order_by = None
    top_n = None
    order_seen = False
    top_seen = False
    condition_groups = []

    segments = _split_top_level_semicolons(spec)
    for segment in segments:
        group_conditions = []
        clauses = _split_top_level_commas(segment)
        for clause in clauses:
            if not clause:
                continue

            if _ORDER_RE.match(clause):
                if order_seen:
                    raise SignalSelectionError(
                        f"Multiple ORDER clauses are not allowed in --signal-selection: {spec!r}"
                    )
                m = _ORDER_FULL_RE.match(clause)
                if not m:
                    raise SignalSelectionError(
                        f"Malformed ORDER clause {clause!r}. Expected: ORDER 'column' [ASC|DESC]"
                    )
                col_raw = m.group("col")
                kind = column_kind(col_raw)
                if kind != "numeric":
                    raise SignalSelectionError(
                        f"ORDER clause references non-numeric column {col_raw!r} — "
                        "only numeric columns are orderable."
                    )
                canonical = _canonical_column_name(col_raw)
                direction = (m.group("dir") or "DESC").upper()
                descending = direction == "DESC"
                order_by = (canonical, descending)
                order_seen = True
                continue

            if _TOP_RE.match(clause):
                if top_seen:
                    raise SignalSelectionError(
                        f"Multiple TOP clauses are not allowed in --signal-selection: {spec!r}"
                    )
                m = _TOP_FULL_RE.match(clause)
                if not m:
                    raise SignalSelectionError(
                        f"Malformed TOP clause {clause!r}. Expected: TOP <int>"
                    )
                n_raw = m.group("n").strip()
                try:
                    n_val = int(n_raw)
                except ValueError:
                    raise SignalSelectionError(
                        f"TOP value must be an integer, got {n_raw!r} in clause {clause!r}"
                    )
                top_n = n_val
                top_seen = True
                continue

            m = _CONDITION_RE.match(clause)
            if not m:
                raise SignalSelectionError(
                    f"Could not parse --signal-selection clause {clause!r}. Expected a condition "
                    "like \"'col' > 1\", an ORDER clause, or a TOP clause."
                )
            col_raw = m.group("col")
            op = m.group("op")
            val_raw = m.group("val")

            if op not in _VALID_OPS:
                raise SignalSelectionError(f"Unknown operator {op!r} in clause {clause!r}")

            kind = column_kind(col_raw)
            canonical = _canonical_column_name(col_raw)
            value = _parse_value(val_raw)

            if kind == "text" and op not in _TEXT_ONLY_OPS:
                raise SignalSelectionError(
                    f"Operator {op!r} cannot be used against text column {col_raw!r} "
                    f"(only {', '.join(sorted(_TEXT_ONLY_OPS))} are valid on text columns) "
                    f"in clause {clause!r}"
                )

            group_conditions.append(SelectionCondition(column=canonical, op=op, value=value))

        condition_groups.append(group_conditions)

    if not condition_groups:
        condition_groups = [[]]
    elif len(condition_groups) > 1:
        for i, group in enumerate(condition_groups):
            if not group:
                raise SignalSelectionError(
                    f"Empty condition group (segment {i + 1} of {len(condition_groups)}, "
                    f"likely a stray ';') in --signal-selection: {spec!r}"
                )

    return SignalSelectionRule(
        conditions=condition_groups[0], order_by=order_by, top_n=top_n, raw=spec,
        condition_groups=condition_groups,
    )


def rule_matches(rule: SignalSelectionRule, candidate, derived: dict, cluster_map: dict):
    """Does `candidate` satisfy `rule`? Handles both the single-group
    (AND-only) case and multi-group OR-of-AND rules uniformly.

    Returns (matched: bool, failing_condition: Optional[SelectionCondition]).
    failing_condition is the condition that caused the FIRST group's
    rejection (for building a diagnostic RULE_FILTERED flag string in
    allocation.py) -- None when matched is True, or when a group happens to
    have zero conditions (vacuously satisfied, nothing to report).
    """
    first_failure = None
    for group in rule.condition_groups:
        group_failure = None
        for cond in group:
            if not evaluate_condition(cond, candidate, derived, cluster_map):
                group_failure = cond
                break
        if group_failure is None:
            return True, None
        if first_failure is None:
            first_failure = group_failure
    return False, first_failure


def evaluate_condition(cond: SelectionCondition, candidate, derived: dict, cluster_map: dict) -> bool:
    """Evaluate a single SelectionCondition against one candidate row."""
    actual = resolve_column(cond.column, candidate, derived, cluster_map)
    op = cond.op
    value = cond.value
    if op == ">":
        return actual > value
    if op == ">=":
        return actual >= value
    if op == "<":
        return actual < value
    if op == "<=":
        return actual <= value
    if op == "==":
        return actual == value
    if op == "!=":
        return actual != value
    raise SignalSelectionError(f"Unknown operator {op!r}")
