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
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta

from kairos.calendars import _DAILY_OR_COARSER

import numpy as np
import pandas as pd

_DB_CONNECT_RETRY_ATTEMPTS = 3
_DB_CONNECT_RETRY_BACKOFF_SECONDS = 1.0


def _connect_with_retry(db_path, attempts=_DB_CONNECT_RETRY_ATTEMPTS,
                         backoff_seconds=_DB_CONNECT_RETRY_BACKOFF_SECONDS):
    """sqlite3.connect() with a short retry-with-backoff on OperationalError.

    A live 6-month papertrade run (2026-07-29) crashed on the very first
    per-date sqlite3.connect(db_path) in run()'s date-major loop with
    "unable to open database file" (SQLITE_CANTOPEN). It did not reproduce
    under a short smoketest, no scheduled kairos systemd timer was active in
    that window, and data/'s directory/file permissions were fine -- so the
    most likely explanation is a transient blip (another concurrent process
    briefly touching pipeline_results.db, or a momentary disk hiccup) rather
    than a deterministic path/permission bug. run()'s date-major loop calls
    connect() once PER DATE (183 times for a 6-month/1d sweep), so a single
    transient failure anywhere in that loop previously killed the entire
    multi-hour run. A short retry is cheap insurance against exactly that,
    without masking a genuinely broken setup -- that still fails on every
    attempt and raises the last exception.
    """
    last_exc = None
    for attempt in range(attempts):
        try:
            return sqlite3.connect(db_path)
        except sqlite3.OperationalError as e:
            last_exc = e
            if attempt < attempts - 1:
                time.sleep(backoff_seconds * (attempt + 1))
    raise last_exc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "data", "pipeline_results.db")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# See _run_group's per-strategy cache precheck for why the key is this fine
# grained (strategy_name, not just group) -- a strategy disabled after a row
# was cached must never be served stale from here; see CLAUDE.md's
# "Signals cache" section.
SIGNALS_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals_cache (
    cache_key TEXT PRIMARY KEY,
    strategy_name TEXT NOT NULL,
    assets TEXT NOT NULL,
    interval TEXT NOT NULL,
    as_of TEXT NOT NULL,
    lookback INTEGER NOT NULL,
    pred_samples INTEGER NOT NULL,
    min_ev_pct REAL NOT NULL,
    model_label TEXT NOT NULL,
    model_path TEXT,
    checkpoint_fingerprint TEXT NOT NULL DEFAULT '',
    stats_json TEXT NOT NULL,
    advice_json TEXT NOT NULL,
    skipped_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


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


def load_accepted_finetuned(conn):
    """Load the accepted-finetuned-model registry.

    Returns {(sorted_assets_csv, interval): model_path} for every
    status='accepted' row in the finetuned_models table (UNIQUE(assets,
    interval) means at most one row per key, so this is a safe dict).

    Returns {} on any sqlite3.Error -- in particular when the
    finetuned_models table doesn't exist yet (fresh clones / older test
    DBs), so callers never need to special-case old databases.
    """
    try:
        cur = conn.execute(
            "SELECT assets, interval, model_path FROM finetuned_models "
            "WHERE status = 'accepted'"
        )
        rows = cur.fetchall()
    except sqlite3.Error:
        return {}
    return {(row[0], row[1]): row[2] for row in rows}


# =============================================================================
# Per-strategy signals cache (signals_cache table in db_path)
# =============================================================================

def _cache_as_of_value(now: datetime, interval: str):
    """Cache-key granularity for `as_of`, per interval.

    Daily-or-coarser bars: whole calendar date, as before -- fetch_data_raw
    only ever consumes as_of.date() for these intervals, so two `now`
    timestamps on the same calendar day fetch identical data and must key
    identically. Intraday bars: floored to the current bar boundary, so a
    freshly-closed bar busts the cache instead of an hour's worth of calls
    all colliding onto one calendar-date key and serving a stale early-day
    snapshot for the rest of the day.
    """
    if interval in _DAILY_OR_COARSER:
        return now.date()
    step = _interval_to_timedelta(interval)
    ref = datetime.min
    return ref + (now - ref) // step * step


def _signals_cache_key(strategy_name, assets_str, interval, as_of_date, lookback,
                        pred_samples, min_ev_pct, model_path, checkpoint_fingerprint) -> str:
    """Canonical cache key for one strategy's rows within one group/pass.

    as_of_date is whatever _cache_as_of_value(now, interval) returned -- a
    bare date for daily-or-coarser intervals (preserving the historical "one
    key per calendar day" behavior), or a bar-floored datetime for intraday
    ones. checkpoint_fingerprint (see
    kairos_strategies._model_checkpoint_fingerprint) busts the cache when a
    finetuned checkpoint is retrained in place at the same model_path,
    mirroring kairos_predcache.make_key's own key design.
    """
    parts = [
        str(strategy_name), str(assets_str), str(interval), as_of_date.isoformat(),
        str(lookback), str(pred_samples), str(min_ev_pct),
        str(model_path or "base"), str(checkpoint_fingerprint or ""),
    ]
    return "|".join(parts)


def _ensure_signals_cache_table(conn):
    conn.executescript(SIGNALS_CACHE_SCHEMA)


def _load_cached_group_result(conn, cache_key):
    """Return (stats_rows, advice_rows, skipped) for cache_key, or None on a miss."""
    row = conn.execute(
        "SELECT stats_json, advice_json, skipped_json FROM signals_cache WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()
    if row is None:
        return None
    stats_json, advice_json, skipped_json = row
    return json.loads(stats_json), json.loads(advice_json), json.loads(skipped_json)


def _store_cached_group_result(conn, cache_key, strategy_name, assets_str, interval,
                                as_of_date, lookback, pred_samples, min_ev_pct,
                                model_label, model_path, checkpoint_fingerprint,
                                stats_rows, advice_rows, skipped):
    """INSERT OR REPLACE so a re-run of the same key overwrites rather than
    duplicating -- row count stays bounded by unique-key space, not call count."""
    conn.execute(
        """INSERT OR REPLACE INTO signals_cache
           (cache_key, strategy_name, assets, interval, as_of, lookback, pred_samples,
            min_ev_pct, model_label, model_path, checkpoint_fingerprint,
            stats_json, advice_json, skipped_json, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (cache_key, strategy_name, assets_str, interval, as_of_date.isoformat(),
         lookback, pred_samples, min_ev_pct, model_label, model_path,
         checkpoint_fingerprint, json.dumps(stats_rows), json.dumps(advice_rows),
         json.dumps(skipped), datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()


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
    "signals_per_week", "model", "asset_class",
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


SIGNALS_COLUMNS = ["ev_pct", "base_win_rate", "signals/backtest", "signal", "model"]
SIGNALS_ALIGN = ["r", "r", "r", "l", "l"]


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
            "model": str(row.get("model", "")),
        })
    return SIGNALS_COLUMNS, SIGNALS_ALIGN, signals_table


def render_report(stats_rows, advice_rows, failures, skipped, timestamp,
                  min_ev_pct=0.10, allocation_section=None,
                  replaced_stats_rows=None) -> str:
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
    replaced_stats_rows: optional list of stats_rows (same shape as
        stats_rows) that were displaced from the base pass by an accepted
        finetuned-model rerun. When non-empty, rendered as a
        "## Replaced base signals (comparison)" section via
        build_stats_table so the base-model numbers stay visible even
        though the finetuned rows replaced them in the main tables.
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

    if replaced_stats_rows:
        lines.append("## Replaced base signals (comparison)")
        lines.append("")
        headers, align, formatted_stats = build_stats_table(replaced_stats_rows)
        lines.extend(format_table(headers, formatted_stats, align))
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
                      credentials_path=None, token_path=None,
                      replaced_stats_rows=None) -> str:
    """Create a new Google Sheet with 'strategies' and 'signals' tabs mirroring
    the markdown report's Stats and Signals tables. Returns the spreadsheet URL.

    First run (or no cached token) opens a browser window for OAuth consent;
    the resulting token is cached to `token_path` for subsequent non-interactive
    runs. See strategy/README.md for one-time Google Cloud setup steps.

    replaced_stats_rows: optional list of stats_rows displaced by an
        accepted finetuned-model rerun; when non-empty, adds a
        'base_shadow' worksheet mirroring the markdown comparison section.
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

    signals_ws = spreadsheet.add_worksheet(
        title="signals", rows=max(len(advice_rows) + 1, 2), cols=len(SIGNALS_COLUMNS))
    if advice_rows:
        headers, _, rows = build_signals_table(advice_rows)
        signals_ws.update([headers] + [[row.get(h, "") for h in headers] for row in rows])
    else:
        signals_ws.update([["No signals generated."]])

    if replaced_stats_rows:
        shadow_ws = spreadsheet.add_worksheet(
            title="base_shadow", rows=max(len(replaced_stats_rows) + 1, 2),
            cols=len(STATS_COLUMNS))
        headers, _, rows = build_stats_table(replaced_stats_rows)
        shadow_ws.update([headers] + [[row.get(h, "") for h in headers] for row in rows])

    return spreadsheet.url


# =============================================================================
# Local spreadsheet export (.xlsx / .ods)
# =============================================================================

SPREADSHEET_ENGINES = {"xlsx": "openpyxl", "ods": "odf"}


def write_spreadsheet(stats_rows, advice_rows, out_path, fmt,
                      allocation_result=None, allocation_config=None,
                      report_date=None, generator_version=None,
                      replaced_stats_rows=None) -> str:
    """Write a spreadsheet ('strategies', 'signals', and optionally 'base_shadow'
    and 'Allocation') to out_path.

    fmt: 'xlsx' or 'ods'. Mirrors the Stats/Signals tables from the markdown
    report and the Google Sheets export (same build_stats_table /
    build_signals_table helpers). When allocation_result and allocation_config
    are provided, adds an 'Allocation' sheet via allocation.py's writer.
    replaced_stats_rows: optional list of stats_rows displaced by an
        accepted finetuned-model rerun; when non-empty, adds a
        'base_shadow' sheet mirroring the markdown comparison section.
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

        if replaced_stats_rows:
            headers, _, rows = build_stats_table(replaced_stats_rows)
            shadow_df = pd.DataFrame(rows, columns=headers)
            shadow_df.to_excel(writer, sheet_name="base_shadow", index=False)

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

def _real_predict_fn(assets_dict, model_path=None):
    """Default predict_fn: batched Kronos prediction (GPU/network required).

    model_path: optional HF repo id or local path for an accepted finetuned
    model (see load_accepted_finetuned); None means the base Kronos model.
    """
    from kairos_strategies import predict_all_batch
    return predict_all_batch(assets_dict, model_path=model_path)


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
        "multi_asset_predictions": multi_preds,
        "current_symbol": symbol,
        "predict_fn": lambda *a, **kw: [],
        "prev_dist": None,
        "bar_index": len(history) - 1,
        "returns_window": returns_window,
        "realized_vol": realized_vol,
    }


def _class_prior_win_rate(interval, assets, model_path, db_path=DB_PATH):
    """Signal-count-weighted mean win rate across ALL strategies in this
    (model, asset class) cell, or None.

    This is the empirical prior `allocation.compute_derived()` shrinks a thin
    strategy toward, instead of a flat 0.5. Returning None (no data, mixed
    group, or any DB problem) restores the previous shrink-toward-0.5
    behaviour exactly, so this can only ever refine sizing, never break it.

    Mixed-class groups return None deliberately: one class's base rate cannot
    stand in for a basket spanning several. Uses plain sqlite3 rather than
    importing kairos_pipeline, matching resolve_disabled_strategies' reasoning
    about the module import cycle.
    """
    from kairos_backtest import asset_class_of_symbol

    classes = {asset_class_of_symbol(a) for a in assets}
    if len(classes) != 1:
        return None
    stage = "finetuned" if model_path else "base"
    clause = "model_path IS NULL" if model_path is None else "model_path = ?"
    params = [interval, next(iter(classes)), stage]
    if model_path is not None:
        params.append(model_path)
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT SUM(win_rate * signal_count) * 1.0 / SUM(signal_count) "
            "FROM strategy_class_stats "
            f"WHERE interval=? AND asset_class=? AND stage=? AND {clause} "
            "AND signal_count > 0",
            params,
        ).fetchone()
        return row[0] if row and row[0] is not None else None
    except sqlite3.Error:
        return None
    finally:
        if conn is not None:
            conn.close()


def _run_group(assets, interval, group_rows, predict_fn, model_path, model_label,
               data, pred_samples, min_ev_pct, conn=None, use_signal_cache=True,
               assets_str=None, as_of_date=None, lookback=None,
               checkpoint_fingerprint=""):
    """Generate stats/advice rows for one (assets, interval) group against
    already-fetched `data`.

    Shared by both the base pass and the accepted-finetuned overlay pass
    (see run()) so the predict -> meta-filter -> generate-signal -> build-row
    pipeline isn't duplicated; only the model used to predict and the label
    stamped onto each row differ between passes.

    model_path: forwarded to predict_fn(data, model_path=...); None for the
        base model.
    model_label: stamped into every stats_row/advice_row's "model" key
        ("Base" or "Finetuned(<assets>)").

    Per-strategy signals cache (signals_cache table, see
    _signals_cache_key): `strategies_by_name` only needs `assets`/`config`
    (built against a dummy predict_fn below), never predict_fn/data -- so
    every strategy's disabled/registry status is resolved *before*
    predict_fn is ever called. A strategy that's since been disabled is
    therefore never a cache read/write candidate; it always falls through
    to the same live "unknown strategy" skip it would hit with no cache at
    all, so a disabled_strategies change always takes effect immediately
    with no stale-cache risk. If every strategy in the group is either
    disabled or already cached, predict_fn is never called at all.
    conn/use_signal_cache/assets_str/as_of_date/lookback/checkpoint_fingerprint
    are only used for cache lookups/writes; conn=None (the default) makes
    caching fully inert regardless of use_signal_cache.

    Returns (stats_rows, advice_rows, skipped) for this group. Raises on
    unexpected errors -- the caller wraps each pass in try/except and
    records group-level failures.
    """
    from kairos_backtest import KairosSettings, Direction
    from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig
    from kairos_strategies import resolve_disabled_strategies

    KairosSettings.interval = interval
    KairosSettings.pred_samples = pred_samples

    stats_rows = []
    advice_rows = []
    skipped = []

    disabled = resolve_disabled_strategies(interval, assets, model_path=model_path)
    class_prior = _class_prior_win_rate(interval, assets, model_path)
    config = OrchestratorConfig.for_interval(interval, disabled_strategies=disabled)

    def _dummy_predict(*a, **kw):
        return []

    orchestrator = KairosOrchestrator(
        predict_fn=_dummy_predict, assets=assets, config=config,
    )
    strategies_by_name = build_strategy_index(orchestrator.strategies)

    cache_active = use_signal_cache and conn is not None
    pending = []  # (row, strat, cache_key)
    for row in group_rows:
        strategy_name = row["strategy_name"]
        # Each viable row targets the group's assets collectively but a
        # signal is generated per-symbol below; try every symbol in the
        # group's asset list and keep whichever fires.
        strat = strategies_by_name.get(strategy_name)
        if strat is None:
            skipped.append(f"{strategy_name}: unknown strategy (not in registry)")
            continue

        cache_key = None
        if cache_active:
            cache_key = _signals_cache_key(
                strategy_name, assets_str, interval, as_of_date, lookback,
                pred_samples, min_ev_pct, model_path, checkpoint_fingerprint,
            )
            cached = _load_cached_group_result(conn, cache_key)
            if cached is not None:
                c_stats, c_advice, c_skipped = cached
                stats_rows.extend(c_stats)
                advice_rows.extend(c_advice)
                skipped.extend(c_skipped)
                continue

        pending.append((row, strat, cache_key))

    if not pending:
        return stats_rows, advice_rows, skipped

    multi_preds = predict_fn(data, model_path=model_path)

    for row, strat, cache_key in pending:
        strategy_name = row["strategy_name"]
        strat_stats = []
        strat_advice = []
        strat_skipped = []

        for sym in assets:
            pred = multi_preds.get(sym)
            if pred is None:
                continue
            dist = pred.dist
            current_price = pred.current_price
            history = pred.history

            if orchestrator._apply_meta_filters(dist, current_price):
                strat_skipped.append(
                    f"{strategy_name}/{sym}: blocked by meta-filters"
                )
                continue

            context = _build_context(orchestrator, sym, current_price, multi_preds, history)

            try:
                sig = strat.generate_signal(dist, current_price, history, context)
            except Exception as e:
                strat_skipped.append(f"{strategy_name}/{sym}: signal generation error ({e})")
                continue

            if sig is None:
                continue

            # Match the backtest's gate (kairos_orchestrator._run_day:
            # `sig.size > 0`): zero-size non-FLAT signals are legit
            # strategy output (Kelly fraction clamped at 0) but never
            # traded, so they must not appear as advice. FLAT signals
            # are exit advice and naturally size 0 — keep them.
            if sig.direction != Direction.FLAT and sig.size <= 0:
                strat_skipped.append(
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
                    strat_skipped.append(
                        f"{strategy_name}/{sym}: ev_pct below threshold "
                        f"({ev_str} < {min_ev_pct:.2f}%)"
                    )
                    continue

            strat_stats.append({
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
                "model": model_label,
                "asset_class": row.get("asset_class"),
                "class_prior_win_rate": class_prior,
            })
            strat_advice.append({
                "expected_value": sig.expected_value,
                "entry": sig.entry,
                "base_win_rate": row.get("base_win_rate"),
                "base_signals": row.get("base_signals"),
                "oracle_signals": row.get("oracle_signals"),
                "signal": signal_to_advice(strategy_name, sym, sig),
                "model": model_label,
            })

        stats_rows.extend(strat_stats)
        advice_rows.extend(strat_advice)
        skipped.extend(strat_skipped)

        if cache_key is not None:
            _store_cached_group_result(
                conn, cache_key, strategy_name, assets_str, interval, as_of_date,
                lookback, pred_samples, min_ev_pct, model_label, model_path,
                checkpoint_fingerprint, strat_stats, strat_advice, strat_skipped,
            )

    return stats_rows, advice_rows, skipped

def run(db_path=DB_PATH, out_dir=RESULTS_DIR, intervals=None, pred_samples=100,
        include_all=False, predict_fn=None, lookback=None, now=None,
        min_ev_pct=0.10, gsheets=False, xlsx=False, ods=False,
        cluster_map_path=None, base_only=False, return_rows=False,
        on_group_timing=None, signal_selection=None, use_signal_cache=True,
        max_leverage=1.0, margin_utilization=0.8,
        margin_config_path="config/margin_ibkr.yaml"):
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
    base_only: if True, skip the accepted-finetuned-model overlay pass
        entirely (every row is labeled "Base", no comparison section/tab).
        Useful for debugging a bad finetuned model, or to force a
        base-only run regardless of the finetuned_models registry.
    return_rows: if True, return (out_path, stats_rows, advice_rows) instead
        of just out_path. Default False preserves the exact old return value
        for every existing caller.
    on_group_timing: optional callback `(assets_str, interval, model_label,
        elapsed_seconds, cache_hit) -> None`, invoked once per (group, pass)
        after each _run_group() call in both the base pass and the
        finetuned-overlay pass. Default None is a true no-op (skips even the
        is_batch_cached() precheck) so existing callers (daily_signals.py,
        weekly_discovery.py, finetune_next) are unaffected. Exists because a
        single date can fan out into many groups x up to 2 passes each, and
        callers timing the whole run() call (e.g. kairos_papertrade.py's
        per-date slow-iteration watchdog) can't tell which group/model
        actually consumed the time, or whether it was a genuine shared-cache
        miss (unexpected once prewarm has run) vs. just many groups adding
        up -- see kairos_papertrade._log_group_timing.
    signal_selection: optional, already-parsed SignalSelectionRule (see
        signal_selection.parse_signal_selection) to pass into AllocationConfig
        as selection_rule, overriding the default min_n/ev_net gate and
        score-sort/top_k ranking in allocation.select_candidates(). Default
        None preserves the exact old default behavior.
    max_leverage / margin_utilization / margin_config_path: forwarded into
        AllocationConfig for Stage 2.5 margin-utilization-target sizing
        (allocation.py). max_leverage > 1.0 (default 1.0, off) enables it;
        existing_margin_used_pct is left at its dataclass default (0.0) --
        this report has no persistent account, always a clean-slate snapshot.
    use_signal_cache: if True (default), cache/reuse each strategy's rows in
        the signals_cache table of db_path, keyed by (strategy, group,
        model+checkpoint, as_of date, lookback, pred_samples, min_ev_pct) --
        see _run_group's docstring and _signals_cache_key. A strategy
        disabled since it was cached is never served stale (checked live,
        before any cache lookup). Set False to always recompute, matching
        pre-cache behavior exactly (also available as --no-signal-cache on
        the CLI).
    """
    from kairos_backtest import KairosSettings, Direction
    from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig
    import kairos_strategies
    from kairos_strategies import (
        fetch_data_raw, resolve_disabled_strategies, LOOKBACK, is_batch_cached,
        _model_checkpoint_fingerprint,
    )

    # Explicit day-boundary clear, on top of predict_all_batch's existing
    # clear-on-model-switch: Pass 1 below calls predict_all_batch with the
    # same model_path=None for every base group in this day, so _dist_cache
    # otherwise accumulates across however many base groups exist before
    # Pass 2's first model switch clears it -- bounded to one day's group
    # count today, but that bound is an emergent property of loop order, not
    # a guarantee. Root-caused 2026-08-13 after the same unbounded-across-
    # dates version of this pattern blew up RSS during prewarm's model-major
    # sweep (see kairos_strategies.predict_all_batch's build_distributions
    # param) -- cheap to make the per-day reset explicit here too rather
    # than rely on run()'s current Pass 1/Pass 2 ordering to keep providing it.
    kairos_strategies._dist_cache.clear()
    kairos_strategies._shared_keys.clear()

    if predict_fn is None:
        predict_fn = _real_predict_fn
    if lookback is None:
        lookback = LOOKBACK
    if now is None:
        now = datetime.now()

    conn = _connect_with_retry(db_path)
    try:
        rows = load_work_items(conn, intervals=intervals, include_all=include_all)
        accepted_finetuned = {} if base_only else load_accepted_finetuned(conn)
        if use_signal_cache:
            _ensure_signals_cache_table(conn)

        groups = group_items(rows)

        group_results = {}  # (assets_str, interval) -> {"stats": [...], "advice": [...]}
        fetched_data_cache = {}  # (assets_str, interval) -> {symbol: DataFrame}
        failures = []
        skipped = []

        # Pass 1: base model, every group.
        for (assets_str, interval), group_rows in groups.items():
            assets = assets_str.split(",")
            try:
                KairosSettings.interval = interval
                KairosSettings.pred_samples = pred_samples

                data = {
                    sym: fetch_data_raw(sym, lookback, as_of=now).tail(lookback)
                    for sym in assets
                }
                fetched_data_cache[(assets_str, interval)] = data

                if on_group_timing is not None:
                    cache_hit = is_batch_cached(data, model_path=None)
                    _group_t0 = time.monotonic()
                group_stats, group_advice, group_skipped = _run_group(
                    assets, interval, group_rows, predict_fn,
                    model_path=None, model_label="Base",
                    data=data, pred_samples=pred_samples, min_ev_pct=min_ev_pct,
                    conn=conn, use_signal_cache=use_signal_cache,
                    assets_str=assets_str, as_of_date=_cache_as_of_value(now, interval),
                    lookback=lookback,
                    checkpoint_fingerprint="",
                )
                if on_group_timing is not None:
                    on_group_timing(assets_str, interval, "Base",
                                     time.monotonic() - _group_t0, cache_hit)
                group_results[(assets_str, interval)] = {
                    "stats": group_stats, "advice": group_advice,
                }
                skipped.extend(group_skipped)
            except Exception as e:
                failures.append(f"group assets={assets_str} interval={interval}: {e}")
                continue

        # Pass 2: overlay accepted-finetuned groups (skipped entirely under
        # --base_only, or when nothing in the registry matches). Reuses pass 1's
        # fetched data (no refetch); displaced base rows move to the
        # replaced_*_rows comparison buckets.
        replaced_stats_rows = []
        replaced_advice_rows = []
        if not base_only and accepted_finetuned:
            for (assets_str, interval), group_rows in groups.items():
                key = (assets_str, interval)
                if key not in group_results:
                    continue  # pass 1 failed this group entirely; nothing to overlay
                sorted_key = ",".join(sorted(assets_str.split(",")))
                model_path = accepted_finetuned.get((sorted_key, interval))
                if model_path is None:
                    continue

                assets = assets_str.split(",")
                try:
                    data = fetched_data_cache[key]
                    model_label = f"Finetuned({assets_str})"
                    if on_group_timing is not None:
                        cache_hit = is_batch_cached(data, model_path=model_path)
                        _group_t0 = time.monotonic()
                    checkpoint_fingerprint = _model_checkpoint_fingerprint(model_path)
                    group_stats, group_advice, group_skipped = _run_group(
                        assets, interval, group_rows, predict_fn,
                        model_path=model_path, model_label=model_label,
                        data=data, pred_samples=pred_samples, min_ev_pct=min_ev_pct,
                        conn=conn, use_signal_cache=use_signal_cache,
                        assets_str=assets_str, as_of_date=_cache_as_of_value(now, interval),
                        lookback=lookback,
                        checkpoint_fingerprint=checkpoint_fingerprint,
                    )
                    if on_group_timing is not None:
                        on_group_timing(assets_str, interval, model_label,
                                         time.monotonic() - _group_t0, cache_hit)
                    # Displace pass 1's base rows for this group into the
                    # comparison buckets, then swap in the finetuned rerun.
                    replaced_stats_rows.extend(group_results[key]["stats"])
                    replaced_advice_rows.extend(group_results[key]["advice"])
                    group_results[key] = {"stats": group_stats, "advice": group_advice}
                    skipped.extend(group_skipped)
                except Exception as e:
                    failures.append(
                        f"group assets={assets_str} interval={interval} (finetuned overlay): {e}"
                    )
                    continue
    finally:
        conn.close()

    stats_rows = []
    advice_rows = []
    for key in groups:
        result = group_results.get(key)
        if result is None:
            continue
        stats_rows.extend(result["stats"])
        advice_rows.extend(result["advice"])

    # Portfolio allocation: derive from structured signal rows when available.
    allocation_result = None
    allocation_config = None
    allocation_section = None
    from allocation import fetch_signals, allocate, AllocationConfig, load_cluster_map, write_md_section
    candidates = fetch_signals(stats_rows, advice_rows)
    if candidates:
        cluster_map = load_cluster_map(cluster_map_path) if cluster_map_path else {}
        ticker_max_leverage = {}
        if max_leverage > 1.0:
            from kairos_margin import load_margin_config, classify_symbol
            margin_cfg = load_margin_config(margin_config_path)
            ticker_max_leverage = {
                c.ticker: 100.0 / classify_symbol(c.ticker, margin_cfg).initial_margin_pct
                for c in candidates
            }
        allocation_config = AllocationConfig(
            cluster_map=cluster_map,
            selection_rule=signal_selection,
            max_leverage=max_leverage,
            margin_utilization_cap=margin_utilization,
            ticker_max_leverage=ticker_max_leverage,
            # Must match kairos_papertrade.py's AllocationConfig construction
            # exactly (see its alloc_config there): gross_cap_pct is a cap on
            # raw notional exposure, not margin -- left at the unleveraged
            # default (100) it clobbers Stage 2.5's margin-utilization target
            # whenever leverage is uneven across selected instruments (real
            # bug found 2026-08-20: 80% margin target scaled down to 8%).
            # This report's whole purpose is to mirror what papertrade would
            # actually do, so it must scale this the same way papertrade does.
            gross_cap_pct=100.0 * max_leverage,
            # existing_margin_used_pct left at its dataclass default (0.0) --
            # deliberate: this report never has a real account, always a
            # clean-slate snapshot.
        )
        allocation_result = allocate(candidates, allocation_config)
        allocation_section = write_md_section(allocation_result, allocation_config)

    os.makedirs(out_dir, exist_ok=True)
    stamp = now.strftime("%Y%m%d%H%M")
    out_path = os.path.join(out_dir, f"kairos_signals_{stamp}.md")
    report = render_report(stats_rows, advice_rows, failures, skipped, now,
                           min_ev_pct=min_ev_pct,
                           allocation_section=allocation_section,
                           replaced_stats_rows=replaced_stats_rows)
    with open(out_path, "w") as f:
        f.write(report)

    if gsheets:
        sheet_url = upload_to_gsheets(stats_rows, advice_rows, now,
                                      replaced_stats_rows=replaced_stats_rows)
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
                replaced_stats_rows=replaced_stats_rows,
            )
            print(sheet_path)

    if return_rows:
        return out_path, stats_rows, advice_rows
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
    gsheets, xlsx, ods, cluster_map_path, base_only, use_signal_cache); `now`
    and `intervals` are set per iteration. use_signal_cache defaulting True
    is exactly what makes repeat --bars_backtest runs over the same window
    fast: each bar's per-strategy rows are cached on first computation.
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
    parser.add_argument("--max-leverage", dest="max_leverage", type=float, default=1.0,
                        help="Maximum leverage for margin-utilization-target sizing "
                             "(default: 1.0, cash-only -- Stage 2.5 in allocation.py "
                             "and the report's Leverage/Margin %% columns stay off).")
    parser.add_argument("--margin-utilization", dest="margin_utilization", type=float, default=0.8,
                        help="Fraction of equity usable as initial margin, target for "
                             "Stage 2.5 sizing (default: 0.8). Only used when "
                             "--max-leverage > 1.0.")
    parser.add_argument("--margin-config", dest="margin_config", default="config/margin_ibkr.yaml",
                        help="Path to YAML margin configuration (default: config/margin_ibkr.yaml). "
                             "Only used when --max-leverage > 1.0.")
    parser.add_argument("--base_only", action="store_true", default=False,
                        help="Skip the accepted-finetuned-model overlay pass "
                             "entirely: every row is labeled 'Base' and no "
                             "comparison section/tab is produced. Useful "
                             "while debugging a bad finetuned model.")
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
    parser.add_argument("--signal-selection", dest="signal_selection", default=None,
                        help="Optional rule string that fully replaces the default "
                             "Portfolio Allocation gating (min_n/positive-EV-net) and "
                             "ranking (score sort, top_k) in allocation.select_candidates(). "
                             "Grammar: comma-separated clauses, each either a condition "
                             "\"'col' OP value\" (OP one of > >= < <= == !=), an "
                             "\"ORDER 'col' [ASC|DESC]\" clause (default DESC, at most one), "
                             "or a \"TOP <int>\" clause (at most one, overrides --top_k-style "
                             "sizing). Column names match the Allocation sheet headers "
                             "(Ticker, Cluster, Strategy, Dir, Entry, Stop, Target, Risk %%, "
                             "Reward %%, b, n, Win raw, Win shrunk, EV raw %%, EV net %%, "
                             "Kelly raw, Score, Sharpe), case-insensitive. Example: "
                             "\"'n' > 60, 'Win raw' > 0.6, ORDER 'EV raw %%' DESC, TOP 3\". "
                             "NOTE: when set, this REPLACES (not adds to) the default "
                             "min_n/EV-positivity gate -- a rule that doesn't check EV "
                             "can admit a negative-EV signal.")
    parser.add_argument("--no-signal-cache", dest="use_signal_cache",
                        action="store_false", default=True,
                        help="Disable the per-strategy signals_cache table in "
                             "db_path (default: enabled). Each strategy's rows "
                             "are normally cached per (strategy, model, group, "
                             "as_of date, lookback, pred_samples, min_ev_pct) so "
                             "a repeat run for an already-computed historical "
                             "bar skips prediction/signal-generation entirely; "
                             "a disabled strategy is never served stale from it. "
                             "Use this flag to always recompute, e.g. while "
                             "debugging.")
    args = parser.parse_args(argv)

    if args.bars_backtest is not None and (not args.intervals or len(args.intervals) != 1):
        parser.error("--bars_backtest requires --intervals to name exactly one interval")

    parsed_signal_selection = None
    if args.signal_selection:
        from signal_selection import parse_signal_selection, SignalSelectionError
        try:
            parsed_signal_selection = parse_signal_selection(args.signal_selection)
        except SignalSelectionError as e:
            parser.error(str(e))

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
            base_only=args.base_only,
            signal_selection=parsed_signal_selection,
            use_signal_cache=args.use_signal_cache,
            max_leverage=args.max_leverage,
            margin_utilization=args.margin_utilization,
            margin_config_path=args.margin_config,
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
        base_only=args.base_only,
        signal_selection=parsed_signal_selection,
        use_signal_cache=args.use_signal_cache,
        max_leverage=args.max_leverage,
        margin_utilization=args.margin_utilization,
        margin_config_path=args.margin_config,
    )
    print(out_path)
    return out_path


if __name__ == "__main__":
    main()
