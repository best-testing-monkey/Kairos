"""kairos_signals.py — Current-signals report generator.

Reads the latest viability_report run from data/pipeline_results.db, groups
viable (strategy, assets, interval) rows by (assets, interval), runs ONE
batched prediction per group, generates a signal per viable strategy against
the latest closed bar, and writes a markdown report to
results/kairos_signals_<YYYYMMDDHHMM>.md.

Structured so the heavy lifting is testable without GPU/network:
  - load_work_items(conn, intervals, include_all)  -- pure DB read
  - group_items(rows)                              -- pure grouping
  - signal_to_advice(strategy_name, symbol, signal) -- pure formatting
  - render_report(...)                             -- pure markdown assembly
  - run(...)                                       -- orchestration, with an
    injectable `predict_fn` so tests can stub out the GPU/network call.
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "data", "pipeline_results.db")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


# =============================================================================
# Pure DB / grouping helpers
# =============================================================================

def load_work_items(conn, intervals=None, include_all=False):
    """Load viability_report rows for the latest run_id.

    Returns a list of dicts (one per row), filtered to `viable=1` unless
    `include_all` is set, and optionally filtered to `intervals`.
    """
    query = "SELECT * FROM viability_report WHERE run_id = (SELECT MAX(run_id) FROM viability_report)"
    params = []
    if not include_all:
        query += " AND viable = 1"
    if intervals:
        placeholders = ",".join("?" for _ in intervals)
        query += f" AND interval IN ({placeholders})"
        params.extend(intervals)

    conn.row_factory = sqlite3.Row
    cur = conn.execute(query, params)
    return [dict(row) for row in cur.fetchall()]


def group_items(rows):
    """Group work-item rows by (assets, interval).

    Returns a dict keyed by (assets_str, interval) -> list of rows, in
    first-seen order (so tests can assert deterministic behavior).
    """
    groups = {}
    for row in rows:
        key = (row["assets"], row["interval"])
        groups.setdefault(key, []).append(row)
    return groups


# =============================================================================
# Advice formatting
# =============================================================================

def _is_missing(value):
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _pct(value, entry):
    if _is_missing(value) or _is_missing(entry) or entry == 0:
        return None
    return (value - entry) / entry * 100.0


def _format_numeric_cell(value, decimals=2):
    """Format a numeric cell with max `decimals` places. None/missing → empty string."""
    if _is_missing(value):
        return ""
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return ""


def _ev_pct_value(expected_value, entry):
    """Expected value as a percentage of entry price, or None if not computable
    (entry <= 0 or either value missing)."""
    if _is_missing(expected_value) or _is_missing(entry) or entry <= 0:
        return None
    try:
        return (float(expected_value) / float(entry)) * 100.0
    except (TypeError, ValueError):
        return None


def _format_ev_pct(expected_value, entry):
    """Format expected value as a percentage of entry price.

    Returns {+/-X.XX%} format, or empty string if entry <= 0 or missing."""
    ev_pct = _ev_pct_value(expected_value, entry)
    if ev_pct is None:
        return ""
    return f"{ev_pct:+.2f}%"


def signal_to_advice(strategy_name, symbol, signal) -> str:
    """Render a single Signal into a plain-English advice bullet (no leading '- ')."""
    from kairos_backtest import Direction

    if signal.direction == Direction.FLAT:
        return f"Strategy {strategy_name} advised **Exit/Flat** on {symbol}."

    direction_word = "Long" if signal.direction == Direction.LONG else "Short"
    size_pct = signal.size * 100.0
    entry = signal.entry

    stop_missing = _is_missing(signal.stop) or signal.stop == 0
    target_missing = _is_missing(signal.target) or signal.target == 0

    if not stop_missing and not target_missing:
        stop_pct = _pct(signal.stop, entry)
        target_pct = _pct(signal.target, entry)
        exit_clause = "Exit by TP/SL."
        return (
            f"Strategy {strategy_name} advised **{direction_word}** position on {symbol} "
            f"for {size_pct:.0f}% liquidity with SL at {signal.stop:,.2f} "
            f"({stop_pct:+.1f}%) and TP at {signal.target:,.2f} ({target_pct:+.1f}%). "
            f"{exit_clause}"
        )
    else:
        exit_clause = f"Exit on {strategy_name} exit signal."
        return (
            f"Strategy {strategy_name} advised **{direction_word}** position on {symbol} "
            f"for {size_pct:.0f}% liquidity. {exit_clause}"
        )


# =============================================================================
# Report rendering
# =============================================================================

def format_table(headers, rows, align):
    """Render a markdown table with proper column width padding and alignment.

    headers: list of column names
    rows: list of dicts (each dict has keys matching headers)
    align: list of "l" or "r" for left/right alignment per column

    Returns list of strings (header, separator, data rows), each padded to
    match column widths so the table aligns in fixed-width text.
    """
    if not headers:
        return []

    # Format all cells
    formatted_rows = []
    for row in rows:
        formatted_row = {}
        for col in headers:
            formatted_row[col] = str(row.get(col, ""))
        formatted_rows.append(formatted_row)

    # Compute column widths: max of header length and any cell length
    col_widths = {}
    for col in headers:
        col_widths[col] = len(col)
        for row in formatted_rows:
            col_widths[col] = max(col_widths[col], len(row.get(col, "")))

    # Build table lines
    lines = []

    # Header row
    header_cells = []
    for col in headers:
        if align[headers.index(col)] == "r":
            header_cells.append(col.rjust(col_widths[col]))
        else:
            header_cells.append(col.ljust(col_widths[col]))
    lines.append("| " + " | ".join(header_cells) + " |")

    # Separator row
    sep_cells = ["-" * col_widths[col] for col in headers]
    lines.append("| " + " | ".join(sep_cells) + " |")

    # Data rows
    for row in formatted_rows:
        row_cells = []
        for col in headers:
            cell = row.get(col, "")
            if align[headers.index(col)] == "r":
                row_cells.append(cell.rjust(col_widths[col]))
            else:
                row_cells.append(cell.ljust(col_widths[col]))
        lines.append("| " + " | ".join(row_cells) + " |")

    return lines


STATS_COLUMNS = [
    "strategy", "symbol", "interval", "backtest_period", "direction", "size",
    "entry", "stop", "target", "expected_value", "ev_pct",
    "oracle_sharpe", "base_sharpe", "oracle_win_rate", "base_win_rate",
    "signals_per_week",
]


def _sort_by_ev_pct_desc(rows):
    """Sort row dicts by ev_pct (expected_value as percent of entry) descending.

    Rows with no computable ev_pct (e.g. FLAT signals) go last. Stable sort:
    ties and missing-value rows keep their insertion order."""
    def key(row):
        ev = _ev_pct_value(row.get("expected_value"), row.get("entry"))
        return (ev is None, -ev if ev is not None else 0.0)
    return sorted(rows, key=key)


def render_report(stats_rows, advice_rows, failures, skipped, timestamp,
                  min_ev_pct=0.10) -> str:
    """Assemble the full markdown report from pre-computed pieces.

    stats_rows: list of dicts with keys from STATS_COLUMNS (only strategies
        that produced >=1 signal should be included by the caller).
    advice_rows: list of dicts with keys:
        - "expected_value": float (2 decimals)
        - "entry": float (for ev_pct calculation)
        - "base_win_rate": float (2 decimals)
        - "base_signals": int or None (number of signals from backtest)
        - "oracle_signals": int or None (fallback if base_signals missing)
        - "signal": plain-English advice string
        Can also be a list of plain strings for backward compatibility (treated as signals).
    failures: list of strings describing group-level failures.
    skipped: list of strings describing skipped/unknown or filtered strategies.
    timestamp: datetime used for the header.
    """
    lines = []
    lines.append(f"# Kairos Signals Report {timestamp.strftime('%Y-%m-%d %H%Mh')}")
    lines.append("")
    lines.append(f"_Filters: min ev_pct {min_ev_pct:.2f}%_")
    lines.append("")
    lines.append("## Stats")
    lines.append("")
    if stats_rows:
        # Format numeric cells in stats table with 2 decimals,
        # rows sorted by ev_pct descending (missing ev_pct last)
        formatted_stats = []
        for row in _sort_by_ev_pct_desc(stats_rows):
            formatted_row = {}
            for col in STATS_COLUMNS:
                if col == "ev_pct":
                    formatted_row[col] = _format_ev_pct(row.get("expected_value"), row.get("entry"))
                elif col in ("size", "entry", "stop", "target", "expected_value",
                           "oracle_sharpe", "base_sharpe", "oracle_win_rate", "base_win_rate",
                           "signals_per_week"):
                    formatted_row[col] = _format_numeric_cell(row.get(col), decimals=2)
                else:
                    formatted_row[col] = str(row.get(col, ""))
            formatted_stats.append(formatted_row)

        # Build stats table with alignment (all numeric columns right-aligned)
        align = []
        for col in STATS_COLUMNS:
            if col in ("size", "entry", "stop", "target", "expected_value", "ev_pct",
                       "oracle_sharpe", "base_sharpe", "oracle_win_rate", "base_win_rate",
                       "signals_per_week"):
                align.append("r")
            else:
                align.append("l")
        table_lines = format_table(STATS_COLUMNS, formatted_stats, align)
        lines.extend(table_lines)
    else:
        lines.append("_No strategies produced a signal in this run._")
    lines.append("")
    lines.append("## Signals")
    lines.append("")
    if advice_rows:
        # Support both new dict format and legacy string format for backward compat
        if advice_rows and isinstance(advice_rows[0], str):
            # Legacy: list of plain strings
            for line in advice_rows:
                lines.append(f"- {line}")
        else:
            # New: list of dicts with ev_pct, base_win_rate, signals/backtest, signal,
            # rows sorted by ev_pct descending (FLAT/missing ev_pct last)
            signals_table = []
            for row in _sort_by_ev_pct_desc(advice_rows):
                ev_pct = _format_ev_pct(row.get("expected_value"), row.get("entry"))
                # signals/backtest: use base_signals, fallback to oracle_signals, blank if both missing
                signals_backtest = ""
                if not _is_missing(row.get("base_signals")):
                    signals_backtest = str(int(row.get("base_signals")))
                elif not _is_missing(row.get("oracle_signals")):
                    signals_backtest = str(int(row.get("oracle_signals")))
                signals_table.append({
                    "ev_pct": ev_pct,
                    "base_win_rate": _format_numeric_cell(row.get("base_win_rate"), decimals=2),
                    "signals/backtest": signals_backtest,
                    "signal": str(row.get("signal", "")),
                })
            signals_headers = ["ev_pct", "base_win_rate", "signals/backtest", "signal"]
            signals_align = ["r", "r", "r", "l"]
            table_lines = format_table(signals_headers, signals_table, signals_align)
            lines.extend(table_lines)
    else:
        lines.append("_No signals generated._")
    lines.append("")

    # Add Legend
    lines.append("### Legend")
    lines.append("")
    lines.append("- `ev_pct` — expected value of the trade per unit, as a percentage of the entry price (probability-weighted over the model's sampled price paths).")
    lines.append("- `base_win_rate` — fraction of winning trades this strategy had in the last base-model backtest.")
    lines.append("- `signals/backtest` — number of signals the strategy generated during the last backtest period; low counts mean win rate and Sharpe are statistically weak.")
    lines.append("")

    if failures:
        lines.append("## Failures")
        lines.append("")
        for f in failures:
            lines.append(f"- {f}")
        lines.append("")

    if skipped:
        lines.append("## Skipped")
        lines.append("")
        for s in skipped:
            lines.append(f"- {s}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Orchestration
# =============================================================================

def _real_predict_fn(assets_dict):
    """Default predict_fn: batched Kronos prediction (GPU/network required)."""
    from kairos_strategies import predict_all_batch
    return predict_all_batch(assets_dict)


def build_strategy_index(strategies):
    """Map every strategy name (wrapper AND inner, down each .base_strategy
    chain) to the OUTERMOST registered instance.

    Most registry entries are wrapper chains (e.g. LiquidityFilterStrategy
    around VaRPositionCap around TrendFollowing); viability_report stores the
    INNER Signal.strategy_name, so the index must resolve inner names.
    Calling generate_signal on the outermost wrapper preserves backtest gating.
    First-seen wins: a later name (wrapper or inner) never overwrites an
    existing exact-match entry.
    """
    index = {}
    for outer in strategies:
        node = outer
        seen_ids = set()
        while node is not None and id(node) not in seen_ids:
            seen_ids.add(id(node))
            name = getattr(node, "name", None)
            if name and name not in index:
                index[name] = outer
            node = getattr(node, "base_strategy", None)
    return index


def _build_context(orchestrator, symbol, current_price, multi_preds, history):
    returns_window = orchestrator._compute_returns_window(
        {sym: pred.history for sym, pred in multi_preds.items()}
    )
    realized_vol = orchestrator._compute_realized_vol(returns_window)
    return {
        "date": history.index[-1],
        "current_price": current_price,
        "capital": orchestrator.capital,
        "multi_asset_predictions": multi_preds,
        "current_symbol": symbol,
        "predict_fn": lambda *a, **kw: [],
        "prev_dist": None,
        "current_position": None,
        "bar_index": len(history) - 1,
        "returns_window": returns_window,
        "realized_vol": realized_vol,
    }


def run(db_path=DB_PATH, out_dir=RESULTS_DIR, intervals=None, pred_samples=100,
        include_all=False, predict_fn=None, lookback=None, now=None,
        min_ev_pct=0.10):
    """Run the full signals-report flow. Returns the path to the written report.

    min_ev_pct: minimum expected value (as percent of entry price) for a
        non-FLAT signal to be reported; lower-EV signals go to the Skipped
        footer. FLAT/exit signals are never filtered. Set 0 to disable.
    """
    from kairos_backtest import KairosSettings, Direction
    from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig
    from kairos_strategies import fetch_data_raw, resolve_disabled_strategies, LOOKBACK

    if predict_fn is None:
        predict_fn = _real_predict_fn
    if lookback is None:
        lookback = LOOKBACK
    if now is None:
        now = datetime.now()

    conn = sqlite3.connect(db_path)
    try:
        rows = load_work_items(conn, intervals=intervals, include_all=include_all)
    finally:
        conn.close()

    groups = group_items(rows)

    stats_rows = []
    advice_rows = []
    failures = []
    skipped = []

    for (assets_str, interval), group_rows in groups.items():
        assets = assets_str.split(",")
        try:
            KairosSettings.interval = interval
            KairosSettings.pred_samples = pred_samples

            data = {
                sym: fetch_data_raw(sym, lookback).tail(lookback)
                for sym in assets
            }

            multi_preds = predict_fn(data)

            disabled = resolve_disabled_strategies(interval, assets)
            config = OrchestratorConfig(disabled_strategies=disabled)

            def _dummy_predict(*a, **kw):
                return []

            orchestrator = KairosOrchestrator(
                predict_fn=_dummy_predict, assets=assets, config=config,
            )
            strategies_by_name = build_strategy_index(orchestrator.strategies)

            for row in group_rows:
                strategy_name = row["strategy_name"]
                # Each viable row targets the group's assets collectively but a
                # signal is generated per-symbol below; try every symbol in the
                # group's asset list and keep whichever fires.
                strat = strategies_by_name.get(strategy_name)
                if strat is None:
                    skipped.append(f"{strategy_name}: unknown strategy (not in registry)")
                    continue

                for sym in assets:
                    pred = multi_preds.get(sym)
                    if pred is None:
                        continue
                    dist = pred.dist
                    current_price = pred.current_price
                    history = pred.history

                    if orchestrator._apply_meta_filters(dist, current_price):
                        skipped.append(
                            f"{strategy_name}/{sym}: blocked by meta-filters"
                        )
                        continue

                    context = _build_context(orchestrator, sym, current_price, multi_preds, history)

                    try:
                        sig = strat.generate_signal(dist, current_price, history, context)
                    except Exception as e:
                        skipped.append(f"{strategy_name}/{sym}: signal generation error ({e})")
                        continue

                    if sig is None:
                        continue

                    # Match the backtest's gate (kairos_orchestrator._run_day:
                    # `sig.size > 0`): zero-size non-FLAT signals are legit
                    # strategy output (Kelly fraction clamped at 0) but never
                    # traded, so they must not appear as advice. FLAT signals
                    # are exit advice and naturally size 0 — keep them.
                    if sig.direction != Direction.FLAT and sig.size <= 0:
                        skipped.append(
                            f"{strategy_name}/{sym}: zero-size signal dropped (no Kelly edge)"
                        )
                        continue

                    # Minimum-EV filter: non-FLAT signals must clear
                    # min_ev_pct (expected value as percent of entry).
                    # FLAT/exit signals are never filtered by this.
                    if sig.direction != Direction.FLAT and min_ev_pct > 0:
                        ev_pct_val = _ev_pct_value(sig.expected_value, sig.entry)
                        if ev_pct_val is None or ev_pct_val < min_ev_pct:
                            ev_str = (f"{ev_pct_val:.2f}%" if ev_pct_val is not None
                                      else "n/a")
                            skipped.append(
                                f"{strategy_name}/{sym}: ev_pct below threshold "
                                f"({ev_str} < {min_ev_pct:.2f}%)"
                            )
                            continue

                    stats_rows.append({
                        "strategy": strategy_name,
                        "symbol": sym,
                        "interval": interval,
                        "backtest_period": row.get("backtest_period"),
                        "direction": sig.direction.name,
                        "size": sig.size,
                        "entry": sig.entry,
                        "stop": sig.stop,
                        "target": sig.target,
                        "expected_value": sig.expected_value,
                        "oracle_sharpe": row.get("oracle_sharpe"),
                        "base_sharpe": row.get("base_sharpe"),
                        "oracle_win_rate": row.get("oracle_win_rate"),
                        "base_win_rate": row.get("base_win_rate"),
                        "signals_per_week": row.get("signals_per_week"),
                    })
                    advice_rows.append({
                        "expected_value": sig.expected_value,
                        "entry": sig.entry,
                        "base_win_rate": row.get("base_win_rate"),
                        "base_signals": row.get("base_signals"),
                        "oracle_signals": row.get("oracle_signals"),
                        "signal": signal_to_advice(strategy_name, sym, sig),
                    })
        except Exception as e:
            failures.append(f"group assets={assets_str} interval={interval}: {e}")
            continue

    os.makedirs(out_dir, exist_ok=True)
    stamp = now.strftime("%Y%m%d%H%M")
    out_path = os.path.join(out_dir, f"kairos_signals_{stamp}.md")
    report = render_report(stats_rows, advice_rows, failures, skipped, now,
                           min_ev_pct=min_ev_pct)
    with open(out_path, "w") as f:
        f.write(report)

    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate a current-signals report from the latest viability run")
    parser.add_argument("--db", default=DB_PATH, help="Path to pipeline_results.db")
    parser.add_argument("--out", default=RESULTS_DIR, help="Output directory for the report")
    parser.add_argument("--intervals", nargs="+", default=None, help="Filter to these intervals")
    parser.add_argument("--pred_samples", type=int, default=100, help="Prediction sample count")
    parser.add_argument("--all", dest="include_all", action="store_true", default=False,
                        help="Include non-viable rows too (default: viable-only)")
    parser.add_argument("--min_ev_pct", type=float, default=0.10,
                        help="Minimum expected value for a signal, in percent of entry "
                             "price (default: 0.10). Non-FLAT signals below this go to "
                             "the Skipped footer; set 0 to disable.")
    args = parser.parse_args(argv)

    out_path = run(
        db_path=args.db, out_dir=args.out, intervals=args.intervals,
        pred_samples=args.pred_samples, include_all=args.include_all,
        min_ev_pct=args.min_ev_pct,
    )
    print(out_path)
    return out_path


if __name__ == "__main__":
    main()
