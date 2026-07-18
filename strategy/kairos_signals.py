#!/usr/bin/env python3
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
import re
import sqlite3
import sys
from datetime import datetime, timedelta

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
    """Load viability_report rows for the latest run_id, per interval.

    Returns a list of dicts (one per row), filtered to `viable=1` unless
    `include_all` is set, and optionally filtered to `intervals`.
    """
    query = (
        "SELECT * FROM viability_report WHERE run_id = ("
        "SELECT MAX(run_id) FROM viability_report v2 "
        "WHERE v2.interval = viability_report.interval)"
    )
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


def build_stats_table(stats_rows):
    """Format stats_rows into (headers, align, formatted_rows) for STATS_COLUMNS.

    stats_rows: list of dicts with keys from STATS_COLUMNS (only strategies
        that produced >=1 signal should be included by the caller).
    Rows are sorted by ev_pct descending (missing ev_pct last), numeric cells
    formatted to 2 decimals.
    """
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

    align = []
    for col in STATS_COLUMNS:
        if col in ("size", "entry", "stop", "target", "expected_value", "ev_pct",
                   "oracle_sharpe", "base_sharpe", "oracle_win_rate", "base_win_rate",
                   "signals_per_week"):
            align.append("r")
        else:
            align.append("l")
    return STATS_COLUMNS, align, formatted_stats


SIGNALS_COLUMNS = ["ev_pct", "base_win_rate", "signals/backtest", "signal"]
SIGNALS_ALIGN = ["r", "r", "r", "l"]


def build_signals_table(advice_rows):
    """Format advice_rows into (headers, align, formatted_rows) for the Signals table.

    advice_rows: list of dicts with keys:
        - "expected_value": float
        - "entry": float (for ev_pct calculation)
        - "base_win_rate": float
        - "base_signals": int or None (number of signals from backtest)
        - "oracle_signals": int or None (fallback if base_signals missing)
        - "signal": plain-English advice string
    Rows are sorted by ev_pct descending (FLAT/missing ev_pct last).
    """
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
    return SIGNALS_COLUMNS, SIGNALS_ALIGN, signals_table


def render_report(stats_rows, advice_rows, failures, skipped, timestamp,
                  min_ev_pct=0.10, allocation_section=None) -> str:
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
    allocation_section: optional markdown string (e.g. from allocation.py
        write_md_section) to append after the Signals section, per RFC §6.
    """
    lines = []
    lines.append(f"# Kairos Signals Report {timestamp.strftime('%Y-%m-%d %H%Mh')}")
    lines.append("")
    lines.append(f"_Filters: min ev_pct {min_ev_pct:.2f}%_")
    lines.append("")
    lines.append("## Stats")
    lines.append("")
    if stats_rows:
        headers, align, formatted_stats = build_stats_table(stats_rows)
        table_lines = format_table(headers, formatted_stats, align)
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
            headers, align, signals_table = build_signals_table(advice_rows)
            table_lines = format_table(headers, signals_table, align)
            lines.extend(table_lines)
    else:
        lines.append("_No signals generated._")
    lines.append("")

    if allocation_section:
        lines.append(allocation_section.rstrip("\n"))
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
# Google Sheets export
# =============================================================================

GSHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]
DEFAULT_GSHEETS_CREDENTIALS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credentials.json")
DEFAULT_GSHEETS_TOKEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "token.json")


def _get_gsheets_credentials(credentials_path, token_path):
    """Load cached OAuth credentials, refreshing or running the first-run
    browser consent flow as needed. Returns a google.oauth2.credentials.Credentials."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GSHEETS_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Google OAuth client secrets not found at {credentials_path}. "
                    "See strategy/README.md 'Google Sheets export' section for setup steps."
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, GSHEETS_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return creds


def upload_to_gsheets(stats_rows, advice_rows, timestamp,
                      credentials_path=None, token_path=None) -> str:
    """Create a new Google Sheet with 'strategies' and 'signals' tabs mirroring
    the markdown report's Stats and Signals tables. Returns the spreadsheet URL.

    First run (or no cached token) opens a browser window for OAuth consent;
    the resulting token is cached to `token_path` for subsequent non-interactive
    runs. See strategy/README.md for one-time Google Cloud setup steps.
    """
    import gspread

    if credentials_path is None:
        credentials_path = DEFAULT_GSHEETS_CREDENTIALS
    if token_path is None:
        token_path = DEFAULT_GSHEETS_TOKEN

    creds = _get_gsheets_credentials(credentials_path, token_path)
    client = gspread.authorize(creds)

    title = f"Kairos Signals {timestamp.strftime('%Y-%m-%d %H%Mh')}"
    spreadsheet = client.create(title)

    strategies_ws = spreadsheet.sheet1
    strategies_ws.update_title("strategies")
    if stats_rows:
        headers, _, rows = build_stats_table(stats_rows)
        strategies_ws.update([headers] + [[row.get(h, "") for h in headers] for row in rows])
    else:
        strategies_ws.update([["No strategies produced a signal in this run."]])

    signals_ws = spreadsheet.add_worksheet(title="signals", rows=max(len(advice_rows) + 1, 2), cols=4)
    if advice_rows:
        headers, _, rows = build_signals_table(advice_rows)
        signals_ws.update([headers] + [[row.get(h, "") for h in headers] for row in rows])
    else:
        signals_ws.update([["No signals generated."]])

    return spreadsheet.url


# =============================================================================
# Local spreadsheet export (.xlsx / .ods)
# =============================================================================

SPREADSHEET_ENGINES = {"xlsx": "openpyxl", "ods": "odf"}


def write_spreadsheet(stats_rows, advice_rows, out_path, fmt,
                      allocation_result=None, allocation_config=None,
                      report_date=None, generator_version=None) -> str:
    """Write a spreadsheet ('strategies', 'signals', and optionally 'Allocation') to out_path.

    fmt: 'xlsx' or 'ods'. Mirrors the Stats/Signals tables from the markdown
    report and the Google Sheets export (same build_stats_table /
    build_signals_table helpers). When allocation_result and allocation_config
    are provided, adds an 'Allocation' sheet via allocation.py's writer.
    Returns out_path.
    """
    engine = SPREADSHEET_ENGINES[fmt]

    if stats_rows:
        headers, _, rows = build_stats_table(stats_rows)
        strategies_df = pd.DataFrame(rows, columns=headers)
    else:
        strategies_df = pd.DataFrame(
            [["No strategies produced a signal in this run."]], columns=["message"])

    if advice_rows:
        headers, _, rows = build_signals_table(advice_rows)
        signals_df = pd.DataFrame(rows, columns=headers)
    else:
        signals_df = pd.DataFrame([["No signals generated."]], columns=["message"])

    with pd.ExcelWriter(out_path, engine=engine) as writer:
        strategies_df.to_excel(writer, sheet_name="strategies", index=False)
        signals_df.to_excel(writer, sheet_name="signals", index=False)

        if allocation_result is not None and allocation_config is not None:
            if engine == "openpyxl":
                from allocation import write_xlsx_sheet
                write_xlsx_sheet(
                    writer.book, allocation_result, allocation_config,
                    report_date=report_date, generator_version=generator_version,
                )
            elif engine == "odf":
                from allocation import write_ods_sheet
                write_ods_sheet(
                    writer.book, allocation_result, allocation_config,
                    report_date=report_date, generator_version=generator_version,
                )

    return out_path


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
        min_ev_pct=0.10, gsheets=False, xlsx=False, ods=False,
        cluster_map_path=None):
    """Run the full signals-report flow. Returns the path to the written report.

    now: the moment treated as "now" — stamps output filenames/report
        headers and caps fetched bars to this moment (rounded down to the
        nearest bar; see fetch_data_raw's `as_of`). Defaults to the real
        current time when not given.
    min_ev_pct: minimum expected value (as percent of entry price) for a
        non-FLAT signal to be reported; lower-EV signals go to the Skipped
        footer. FLAT/exit signals are never filtered. Set 0 to disable.
    gsheets: if True, also upload the Stats/Signals tables to a new Google
        Sheet (see upload_to_gsheets); the sheet URL is printed to stdout.
    xlsx / ods: if True, also write the Stats/Signals tables to a local
        kairos_signals_<stamp>.xlsx / .ods file in out_dir (see
        write_spreadsheet); the path is printed to stdout.
    cluster_map_path: optional path to a CSV file mapping ticker -> cluster
        name for the portfolio allocation sheet/section.
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
                sym: fetch_data_raw(sym, lookback, as_of=now).tail(lookback)
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

    # Portfolio allocation: derive from structured signal rows when available.
    allocation_result = None
    allocation_config = None
    allocation_section = None
    from allocation import fetch_signals, allocate, AllocationConfig, load_cluster_map, write_md_section
    candidates = fetch_signals(stats_rows, advice_rows)
    if candidates:
        cluster_map = load_cluster_map(cluster_map_path) if cluster_map_path else {}
        allocation_config = AllocationConfig(cluster_map=cluster_map)
        allocation_result = allocate(candidates, allocation_config)
        allocation_section = write_md_section(allocation_result, allocation_config)

    os.makedirs(out_dir, exist_ok=True)
    stamp = now.strftime("%Y%m%d%H%M")
    out_path = os.path.join(out_dir, f"kairos_signals_{stamp}.md")
    report = render_report(stats_rows, advice_rows, failures, skipped, now,
                           min_ev_pct=min_ev_pct,
                           allocation_section=allocation_section)
    with open(out_path, "w") as f:
        f.write(report)

    if gsheets:
        sheet_url = upload_to_gsheets(stats_rows, advice_rows, now)
        print(sheet_url)

    report_date = now.strftime("%Y-%m-%d")
    generator_version = "kairos_signals/0.1.0"
    for fmt, enabled in (("xlsx", xlsx), ("ods", ods)):
        if enabled:
            sheet_path = os.path.join(out_dir, f"kairos_signals_{stamp}.{fmt}")
            write_spreadsheet(
                stats_rows, advice_rows, sheet_path, fmt,
                allocation_result=allocation_result,
                allocation_config=allocation_config,
                report_date=report_date,
                generator_version=generator_version,
            )
            print(sheet_path)

    return out_path


_INTERVAL_UNIT_TIMEDELTA = {
    "m": lambda n: timedelta(minutes=n),
    "h": lambda n: timedelta(hours=n),
    "d": lambda n: timedelta(days=n),
    "wk": lambda n: timedelta(weeks=n),
}


def _interval_to_timedelta(interval: str) -> timedelta:
    """Convert an interval string (e.g. "1d", "1h", "60m", "30m", "1wk") to a
    fixed timedelta bar size. Calendar-based units ("1mo", "3mo") have no
    fixed duration and are not supported."""
    match = re.fullmatch(r"(\d+)(mo|wk|d|h|m)", interval)
    if not match or match.group(2) == "mo":
        raise ValueError(f"Cannot convert interval {interval!r} to a fixed timedelta step")
    count, unit = int(match.group(1)), match.group(2)
    return _INTERVAL_UNIT_TIMEDELTA[unit](count)


def run_bars_backtest(base_now, interval, bars_backtest, **run_kwargs) -> list:
    """Generate `bars_backtest` signals reports, one per bar of `interval`,
    stepping backward from `base_now` (the most recent report) to
    `base_now - (bars_backtest - 1) * bar_size` (the oldest).

    `run_kwargs` is forwarded to each `run()` call unchanged (db_path,
    out_dir, pred_samples, include_all, predict_fn, lookback, min_ev_pct,
    gsheets, xlsx, ods, cluster_map_path); `now` and `intervals` are set per
    iteration.
    """
    step = _interval_to_timedelta(interval)
    out_paths = []
    for i in range(bars_backtest):
        iter_now = base_now - i * step
        out_paths.append(run(now=iter_now, intervals=[interval], **run_kwargs))
    return out_paths


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
    parser.add_argument("--gsheets", action="store_true", default=False,
                        help="Also upload the Stats/Signals tables to a new Google Sheet "
                             "(tabs 'strategies' and 'signals'). First run requires "
                             "one-time OAuth setup, see strategy/README.md.")
    parser.add_argument("--xlsx", action="store_true", default=False,
                        help="Also write the Stats/Signals tables to a local "
                             "kairos_signals_<stamp>.xlsx file (no setup required).")
    parser.add_argument("--ods", action="store_true", default=False,
                        help="Also write the Stats/Signals tables to a local "
                             "kairos_signals_<stamp>.ods file (no setup required).")
    parser.add_argument("--cluster_map", default=None,
                        help="Optional path to a CSV file mapping ticker -> "
                             "cluster name for the Allocation sheet/section.")
    parser.add_argument("--effective_per", default=None,
                        help='Treat this moment as "now": \'YYYYMMDD [HHnn]\' '
                             '(e.g. "20260615 1430" or "20260615"; time '
                             'defaults to 0000). Caps fetched bars to this '
                             'moment (rounded down to the nearest bar) and stamps '
                             'report/filenames with it, instead of the real '
                             'current time. Useful for backtesting/QA the report.')
    parser.add_argument("--bars_backtest", type=int, default=None,
                        help='Generate N reports, one per bar of --intervals '
                             '(required to be a single interval), stepping '
                             'backward from --effective_per (or now) as the '
                             'most recent report. E.g. "--bars_backtest 28" '
                             '-> 28 reports for the past 28 bars.')
    args = parser.parse_args(argv)

    if args.bars_backtest is not None and (not args.intervals or len(args.intervals) != 1):
        parser.error("--bars_backtest requires --intervals to name exactly one interval")

    now = None
    if args.effective_per is not None:
        fmt = "%Y%m%d %H%M" if " " in args.effective_per else "%Y%m%d"
        now = datetime.strptime(args.effective_per, fmt)

    if args.bars_backtest is not None:
        base_now = now if now is not None else datetime.now()
        out_paths = run_bars_backtest(
            base_now, args.intervals[0], args.bars_backtest,
            db_path=args.db, out_dir=args.out,
            pred_samples=args.pred_samples, include_all=args.include_all,
            min_ev_pct=args.min_ev_pct, gsheets=args.gsheets,
            xlsx=args.xlsx, ods=args.ods,
            cluster_map_path=args.cluster_map,
        )
        for p in out_paths:
            print(p)
        return out_paths

    out_path = run(
        db_path=args.db, out_dir=args.out, intervals=args.intervals,
        pred_samples=args.pred_samples, include_all=args.include_all,
        min_ev_pct=args.min_ev_pct, gsheets=args.gsheets,
        xlsx=args.xlsx, ods=args.ods, now=now,
        cluster_map_path=args.cluster_map,
    )
    print(out_path)
    return out_path


if __name__ == "__main__":
    main()
