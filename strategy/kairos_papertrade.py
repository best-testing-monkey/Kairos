#!/usr/bin/env python3
"""kairos_papertrade.py — Paper-trade executor (roadmap Phase 4.1).

Replays a window of `kairos_signals.py` reports through Phantom Ledger
(package `phantom-ledger`, imported as `phantom`), a sibling paper-trading
engine, applying a ONE-REPORT LAG so that candidates recommended by report
`i` execute at report `i+1`'s date (next-bar open) -- see
roadmap/phase-4-paper-trading.md, "Every recommendation is 'executed' at
next-bar open."

Structured so the pure logic is unit-testable without a live Phantom/GPU
install:
  - parse_report_effective_dt(report_path) -- header-line regex parse
  - generate_and_dedupe_reports(...)       -- report generation + de-dup
  - map_instrument_type(...)                -- ticker/direction -> stock|cfd
  - compute_pct_profit_per_trade(...)        -- pure P&L math
  - write_json_report(...)                   -- JSON shape
The live Phantom Ledger loop (main()) requires the `phantom` package and
historical price data; it is smoke-tested manually (see task notes), not
covered by the automated unit-test file.

HISTORICAL NOTE: phantom_ledger's SimulationEngine used to only fetch price
bars for `tickers[0]` of a multi-ticker `runner.backtest()` call and apply
that one bar to every order/position regardless of its own ticker (verified
live: a second ticker's order filled at the first ticker's price). This was
fixed upstream in phantom_ledger commit 9e36be102bb59e77655adba2aba2dba49272c3f8
(SimulationEngine now fetches bars per-ticker and marks each position to its
own ticker's price), so the day-by-day loop below makes one plain combined
`runner.backtest(tickers=sorted(open_tickers | new_tickers), ...)` call per
day again, as originally designed -- no client-side per-ticker workaround
needed.
"""
import argparse
import gc
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Optional

import sqlite3

import pandas as pd
import price_cache
from tqdm import tqdm
from sqlitedict import SqliteDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kairos_signals import DB_PATH, RESULTS_DIR, _interval_to_timedelta
import kairos_signals as _kairos_signals_mod
from kairos.ops import GpuLock, OpsError, send_telegram
import kairos_strategies

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PHANTOM_DATA_DIR = os.path.join(REPO_ROOT, "data", "phantom_ledger")
DEFAULT_PRED_CACHE_DIR = os.path.join(REPO_ROOT, "data", "predcache")
WATCHDOG_LOG_PATH = os.path.join(REPO_ROOT, "data", "papertrade_watchdog.log")

# Finest-to-coarsest intraday intervals to try, in order, before falling
# back to phantom's own daily ("1d") behavior. 3h/12h were requested but
# price_cache/yfinance-style intervals don't support them (kairos/data.py's
# `_SUPPORTED_INTERVALS` has no 3h/12h entry) -- only 1m/15m/30m/1h/1d are
# real options, so the ladder is 1m -> 15m -> 30m -> 1h -> 1d.
_INTRADAY_FALLBACK_LADDER = ["1m", "15m", "30m", "1h"]

# Watchdog threshold for per-iteration Telegram notifications during the
# long-running report-generation and day-by-day backtest loops (a single
# iteration/day taking this long is treated as an outlier worth a heads-up,
# not a full every-iteration spam).
_SLOW_ITERATION_THRESHOLD_SECONDS = 60.0

# Per-(group, pass) threshold for the cheaper, subprocess-free companion log
# (_log_group_timing). Deliberately much smaller than the per-date threshold
# above: a single date's run() call can fan out into dozens of groups x up
# to 2 passes each sharing that 300s budget, so an individual group taking
# even 30s is worth a line -- see _log_group_timing's docstring.
_SLOW_GROUP_THRESHOLD_SECONDS = 30.0

# prewarm_prediction_cache's check/load loops call gc.collect() every this
# many iterations. _materialize_model() is the only other gc.collect() call
# site in this codebase, and it only fires on a model switch -- the base
# sweep processes ~150 groups x ~183 dates against ONE model with no switch
# at all, so without this, reference-cycle garbage from thousands of
# transient DataFrames can accumulate uncollected for the sweep's entire
# duration (CPython's generational GC thresholds are allocation-COUNT based,
# not memory-size based, so large-but-few pandas objects are exactly the
# case it's slow to notice). Contributed to the 2026-07-29 leak.
_PREWARM_GC_INTERVAL = 500

# Mirrors the private `_configured`/`_ensure_configured` pattern in
# phantom/data/yahoo.py (not importable -- it's private) so we only call
# price_cache.configure() once per db_path.
_configured_dbs: set[str] = set()


def _ensure_configured_db(db_path: str) -> None:
    if db_path not in _configured_dbs:
        price_cache.configure(remote=False, local_mirror_path=db_path)
        _configured_dbs.add(db_path)


def _notify(text: str, enabled: bool = True) -> None:
    """Send a Telegram message, never letting a notification failure crash
    the caller.

    Catches OpsError (missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID
    credentials, or a Telegram API failure) and prints a warning to stderr
    instead of raising. Credentials are read from the environment by
    kairos.ops.send_telegram itself -- see .env.example (loaded from
    ~/.config/kairos/kairos.env in production) for the documented source.
    `enabled=False` (--no-telegram on the CLI) is a silent no-op.

    Sends with `parse_mode=None` (plain text): these messages embed dynamic,
    uncontrolled content (asset symbols, stderr/traceback tails), and a
    single unbalanced Markdown special character anywhere in that content --
    including the literal underscore in "finetune_next" -- makes Telegram's
    legacy Markdown parser reject the whole message with a 400 "can't parse
    entities" error (see CLAUDE.md's "Telegram notifications" section).
    Plain text can never fail to parse.
    """

    print(text)

    if not enabled:
        return
    try:
        send_telegram(text, parse_mode=None)
    except OpsError as exc:
        print(f"[kairos_papertrade] WARNING: Telegram notification failed: {exc}", file=sys.stderr)


def _read_self_rss_kb() -> Optional[int]:
    """Current resident set size of THIS process, in KiB, via /proc/self/status.

    Deliberately reads /proc directly instead of shelling out (unlike the
    free/nvidia-smi snapshots below) -- it's a per-process, per-call figure we
    want cheaply and reliably on every watchdog fire, not just best-effort.
    Returns None off-Linux or if /proc is unavailable.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])  # kB
    except (OSError, ValueError, IndexError):
        pass
    return None


def _log_watchdog_snapshot(context: str, elapsed: float) -> None:
    """Append a timestamped forensic snapshot to WATCHDOG_LOG_PATH when the
    slow-iteration watchdog fires (see _SLOW_ITERATION_THRESHOLD_SECONDS's
    two call sites).

    This is deliberately logging-only, NOT a fix: an overnight papertrade
    run froze the machine (2026-07-28/29) and the leading hypothesis at the
    time this was written was a RAM/swap brownout, but the math doesn't
    support the prediction cache as the culprit and this machine has a
    separate, already-documented history of GPU/display driver hangs that
    matches the symptom (full freeze, no clean shutdown) at least as well.
    Rather than guess-fixing an unconfirmed cause, this captures `free -h`
    and `nvidia-smi` output at the moment a slow iteration is detected, so a
    future freeze has forensic evidence instead of nothing. Never touches
    kairos_predcache's mem_fraction/LRU sizing -- out of scope, see above.

    A second freeze (2026-07-29) recurred with the machine-wide RAM climbing
    steadily over ~90 minutes (per `sar -r`) while this log's own entries
    showed only date context, no PID -- indistinguishable from either an
    in-process leak or two overlapping `kairos_papertrade.py` invocations
    sharing this same log file (bash history that night showed the exact
    same command launched twice). PID + this process's own VmRSS are logged
    now so the next occurrence settles which it is: multiple distinct PIDs
    in the log around the same time means overlapping runs; a single PID
    whose own RSS climbs in step with system-wide usage means a real
    in-process leak.

    Best-effort only: every subprocess call and the log write itself are
    wrapped in try/except so a missing binary (e.g. no nvidia-smi on a
    CPU-only box), a subprocess timeout, or a log-write failure can never
    crash or block the actual backtest loop.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    rss_kb = _read_self_rss_kb()
    rss_str = f"{rss_kb / (1024 * 1024):.2f}GiB" if rss_kb is not None else "unknown"
    lines = [
        f"=== {timestamp} slow iteration: {context} (elapsed={elapsed:.1f}s) "
        f"pid={os.getpid()} self_rss={rss_str} ==="
    ]

    try:
        free_result = subprocess.run(
            ["free", "-h"], capture_output=True, text=True, timeout=10,
        )
        lines.append(free_result.stdout)
    except Exception as exc:
        lines.append(f"[free -h failed: {exc}]")

    try:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv"],
            capture_output=True, text=True, timeout=10,
        )
        lines.append(gpu_result.stdout)
    except Exception as exc:
        lines.append(f"[nvidia-smi failed: {exc}]")

    try:
        os.makedirs(os.path.dirname(WATCHDOG_LOG_PATH), exist_ok=True)
        with open(WATCHDOG_LOG_PATH, "a") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as exc:
        print(f"[kairos_papertrade] WARNING: watchdog log write failed: {exc}", file=sys.stderr)


def _log_group_timing(iter_now, assets_str: str, interval: str, model_label: str,
                       elapsed: float, cache_hit: bool) -> None:
    """Cheap, subprocess-free companion to _log_watchdog_snapshot.

    generate_and_dedupe_reports's per-date watchdog only measures the whole
    kairos_signals.run() call for that date -- a date can fan out into dozens
    of (assets, interval) groups x up to 2 passes (base + finetuned overlay)
    each, so a >5min date-level entry can't say which group/model actually
    consumed the time, or whether prewarm's shared-cache coverage held up
    during report generation. Wired in as run()'s on_group_timing callback
    (see kairos_signals.run's docstring) so every individual group/pass that
    was either slow (> _SLOW_GROUP_THRESHOLD_SECONDS) or a shared-cache MISS
    (unexpected once prewarm has already run for this model/date) gets one
    line here -- with no subprocess calls, so unlike _log_watchdog_snapshot
    it's cheap enough to call once per group without adding real overhead to
    the report-generation loop itself.

    Best-effort: a log-write failure is swallowed (never blocks report
    generation), matching _log_watchdog_snapshot's contract.
    """
    timestamp = datetime.now().isoformat(timespec="seconds")
    status = "HIT" if cache_hit else "MISS"
    line = (
        f"{timestamp} group_timing date={iter_now:%Y-%m-%d} pid={os.getpid()} "
        f"model={model_label} interval={interval} assets={assets_str} "
        f"elapsed={elapsed:.1f}s cache={status}\n"
    )
    try:
        os.makedirs(os.path.dirname(WATCHDOG_LOG_PATH), exist_ok=True)
        with open(WATCHDOG_LOG_PATH, "a") as f:
            f.write(line)
    except Exception as exc:
        print(f"[kairos_papertrade] WARNING: group-timing log write failed: {exc}", file=sys.stderr)


def _make_group_timing_cb(iter_now):
    """Build the on_group_timing callback passed to kairos_signals.run() for
    a given date, closing over `iter_now` so _log_group_timing's entries
    carry the right date without threading it through run()'s callback
    signature. Only logs groups that are slow or a cache miss -- a fully
    warm, fast group is exactly the expected case and not worth a line."""
    def _cb(assets_str, interval, model_label, elapsed, cache_hit):
        if not cache_hit or elapsed > _SLOW_GROUP_THRESHOLD_SECONDS:
            _log_group_timing(iter_now, assets_str, interval, model_label, elapsed, cache_hit)
    return _cb


class _IntradayFallbackProvider:
    """Wraps phantom's HistoricalProvider, trying finer intervals first for
    get_bars() so order fills/TP/SL evaluate against real intraday bars
    when available, falling back to phantom's own daily behavior otherwise.
    Only affects fill/TP/SL evaluation inside runner.backtest() -- report
    generation cadence (--interval) is untouched."""

    def __init__(self, data_dir):
        from phantom.data.yahoo import HistoricalProvider

        self._fallback = HistoricalProvider(data_dir)
        self._db_path = str(Path(data_dir) / "yfd_prices.db")

    def get_bars(self, ticker, start, end) -> pd.DataFrame:
        _ensure_configured_db(self._db_path)
        for interval in _INTRADAY_FALLBACK_LADDER:
            try:
                df = price_cache.get_price_data(
                    ticker,
                    start.strftime("%Y-%m-%d"),
                    end.strftime("%Y-%m-%d"),
                    interval=interval,
                    db_path=self._db_path,
                )
            except Exception as exc:
                print(
                    f"WARNING: intraday fetch failed for {ticker} at "
                    f"interval={interval}: {exc}", file=sys.stderr,
                )
                continue

            if df is None or df.empty:
                continue

            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            if df.index.tz is None:
                df.index = df.index.tz_localize("America/New_York")
            df.index = df.index.tz_convert("UTC")

            start_ts = pd.Timestamp(start, tz="UTC")
            end_ts = pd.Timestamp(end, tz="UTC")
            sliced = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
            if not sliced.empty:
                return sliced
            # Empty after slicing to [start, end] -- try the next, coarser
            # interval rather than returning an empty frame early.

        return self._fallback.get_bars(ticker, start, end)

    def get_current_price(self, ticker):
        return self._fallback.get_current_price(ticker)

    def get_bid_ask(self, ticker):
        return self._fallback.get_bid_ask(ticker)

    def get_dividends(self, ticker, start, end):
        return self._fallback.get_dividends(ticker, start, end)


# =============================================================================
# Pure helpers (unit-testable without a live `phantom` install)
# =============================================================================

_HEADER_RE = re.compile(r"^# Kairos Signals Report (\d{4}-\d{2}-\d{2} \d{4}h)$")


def parse_report_effective_dt(report_path):
    """Parse the true effective datetime from a report's FIRST LINE.

    Never trusts the filename or file mtime -- kairos_signals.py's own
    report header (`# Kairos Signals Report YYYY-MM-DD HHMMh`) is the only
    source of truth for "when" a report was generated as-of.

    Raises ValueError if the first line doesn't match the expected header
    format (rather than silently falling back to anything else).
    """
    with open(report_path, "r") as f:
        first_line = f.readline().rstrip("\n")
    m = _HEADER_RE.match(first_line)
    if not m:
        raise ValueError(
            f"Report {report_path!r} has an unexpected header line "
            f"({first_line!r}); refusing to guess its effective datetime."
        )
    return datetime.strptime(m.group(1), "%Y-%m-%d %H%Mh")


def generate_and_dedupe_reports(base_now, interval, months_back, run_kwargs, notify: bool = True):
    """Generate a window of kairos_signals reports, stepping backward from
    `base_now`, and de-dupe by each report's true effective datetime.

    Steps `iter_now` back by `_interval_to_timedelta(interval)` for
    `round(months_back * 30.44 / days_per_step)` iterations (for "1d" that's
    just `round(months_back * 30.44)` iterations), calling
    `kairos_signals.run(now=iter_now, intervals=[interval], return_rows=True,
    **run_kwargs)` each time. De-dupes by the report's parsed effective_dt
    (first-seen wins -- e.g. weekend/holiday reports that all resolve to the
    same last-closed-bar date).

    Watchdog: times each `kairos_signals.run()` call. If a single call takes
    longer than `_SLOW_ITERATION_THRESHOLD_SECONDS` (5 minutes), sends a
    Telegram heads-up via `_notify` (enabled=`notify`) so a run that's stuck
    or unusually slow (e.g. a finetuned-overlay pass) is visible without
    spamming a message per iteration -- only outliers notify. Each `run()`
    call also gets a per-date on_group_timing callback (see
    _make_group_timing_cb/_log_group_timing) so a slow date's *groups* are
    individually visible in data/papertrade_watchdog.log, not just the
    date-level aggregate.

    Returns a list of (effective_dt, stats_rows, advice_rows) tuples, sorted
    oldest-first.
    """
    step = _interval_to_timedelta(interval)
    days_per_step = step.total_seconds() / 86400.0
    n_iterations = round(months_back * 30.44 / days_per_step)
    base_now = floor_dt(base_now,interval=step);

    hash_v2, hash_legacy = _make_report_hash(base_now, interval, run_kwargs)
    # todo: install SqliteDict un uv (and lockfile or something.. don't know how that works with u)
    #   d = SqliteDict('mydata.sqlite', autocommit=True)
    #   d['key'] = {'some': 'value'}
    #   d.close()

    seen_table = _pick_seen_table("report_seen.db", hash_v2, hash_legacy)
    seen = SqliteDict(filename="report_seen.db", tablename=seen_table, autocommit=True)
    for i in tqdm(iterable=range(n_iterations), desc="Interval report", unit="bar"):
        generate_and_dedupe_report_single(seen, base_now, i, step, interval, run_kwargs, n_iterations, notify)

    return [seen[key] for key in sorted(seen.keys())]


def json_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _make_report_hash(base_now, interval, run_kwargs):
    """Return (v2_hash, legacy_hash) for the report de-dup DB table name.

    v2 includes accepted-finetuned model paths in the key, so a newly
    accepted finetuned model for an existing group busts report resume.
    legacy is the pre-2026-08-03 hash that only covered base_now/interval/
    work-item groups; it is kept as a read fallback so in-flight runs do
    not suddenly see empty tables and regenerate everything.
    """
    db_path = run_kwargs.get("db_path", DB_PATH)
    base_only = run_kwargs.get("base_only", False)

    # See kairos_signals._connect_with_retry's docstring for why this isn't
    # a plain sqlite3.connect() -- a live run crashed here (and in run()'s
    # date-major loop) with a transient "unable to open database file".
    conn = _kairos_signals_mod._connect_with_retry(db_path)
    try:
        rows = _kairos_signals_mod.load_work_items(conn, intervals=[interval])
        groups = _kairos_signals_mod.group_items(rows)
        accepted_finetuned = (
            {} if base_only else _kairos_signals_mod.load_accepted_finetuned(conn)
        )
    finally:
        conn.close()

    # Legacy hash: identical to the pre-fix computation so existing
    # seen_<hash> tables continue to be found.
    legacy_key = [base_now, interval, "base"]
    legacy_key += groups
    legacy_hash = hashlib.sha256(
        json.dumps(legacy_key, sort_keys=True, default=json_default).encode()
    ).hexdigest()

    # v2 hash: additionally folds in accepted-finetuned model paths.
    v2_key = [base_now, interval, "base", "v2"]
    v2_key += sorted(groups)
    if not base_only and accepted_finetuned:
        finetuned = []
        for (assets_str, grp_interval), _group_rows in groups.items():
            sorted_key = ",".join(sorted(assets_str.split(",")))
            model_path = accepted_finetuned.get((sorted_key, grp_interval))
            if model_path is not None:
                finetuned.append(model_path)
        v2_key.append(sorted(finetuned))
    v2_hash = hashlib.sha256(
        json.dumps(v2_key, sort_keys=True, default=json_default).encode()
    ).hexdigest()

    return v2_hash, legacy_hash


def _table_has_rows(db_path: str, table_name: str) -> bool:
    """Best-effort row count check for _pick_seen_table. Treats any sqlite
    error as "no rows" so a missing/corrupt DB never blocks the run."""
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            if cur.fetchone()[0] == 0:
                return False
            cur = conn.execute(f'SELECT count(*) FROM "{table_name}"')
            return cur.fetchone()[0] > 0
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def _pick_seen_table(seen_db_path: str, hash_v2: str, hash_legacy: str) -> str:
    """Choose which seen table to open.

    Prefer a populated v2 table; fall back to a populated legacy table so
    existing in-flight runs resume without a full regen. If neither exists,
    start fresh with v2.
    """
    v2_table = f"seen_v2_{hash_v2}"
    legacy_table = f"seen_{hash_legacy}"
    if _table_has_rows(seen_db_path, v2_table):
        return v2_table
    if _table_has_rows(seen_db_path, legacy_table):
        return legacy_table
    return v2_table


def generate_and_dedupe_report_single(seen, base_now: datetime, i:int, step: timedelta, interval, run_kwargs, n_iterations, notify):
    iter_now = base_now - i * step

    if iter_now in seen:
        return seen[iter_now]

    start_t = time.monotonic()
    out_path, stats_rows, advice_rows = _kairos_signals_mod.run(
        now=iter_now, intervals=[interval], return_rows=True,
        on_group_timing=_make_group_timing_cb(iter_now), **run_kwargs
    )
    elapsed = time.monotonic() - start_t
    if elapsed > _SLOW_ITERATION_THRESHOLD_SECONDS:
        _notify(
            f"⏱️ Kairos papertrade: report {i + 1}/{n_iterations} "
            f"(date {iter_now:%Y-%m-%d}) took {elapsed / 60:.1f}min (>5min) — "
            f"still running",
            enabled=notify,
        )
        _log_watchdog_snapshot(
            f"report {i + 1}/{n_iterations} (date {iter_now:%Y-%m-%d})", elapsed,
        )
    if iter_now not in seen:
        seen[iter_now] = (iter_now, stats_rows, advice_rows)

def _format_prewarm_load_message(model_label: str, dates: list) -> str:
    """🧠 pre-load heads-up notification text, sent right before
    prewarm_prediction_cache triggers a real Kronos model load for one
    sweep unit (the base sweep, or one finetuned group's sweep) -- i.e.
    only when that unit has at least one genuine kairos_predcache miss.

    Distinct emoji from this file's other notifications (🟢 start, ✅
    success, ⚠️ soft-fail, ❌ hard failure, 💥 crash, ⏱️ slow-iteration
    watchdog) so a model-load heads-up reads differently from those."""
    start = min(dates).strftime("%Y-%m-%d")
    end = max(dates).strftime("%Y-%m-%d")
    return (
        f"🧠 Kairos papertrade prewarm: loading {model_label} model — "
        f"period {start} → {end} ({len(dates)} dates)"
    )


def _ensure_pred_cache_dir_env() -> str:
    """Point KAIROS_PRED_CACHE_DIR at a persistent, project-local cache dir
    for this run, unless the caller already set it.

    Persistent, project-local shared prediction cache dir (data/predcache/
    -- NOT /tmp/tmpfs, survives across invocations and reboots; see
    DEFAULT_PRED_CACHE_DIR and CLAUDE.md's "Model-major prediction prewarm"
    section). If the caller already set KAIROS_PRED_CACHE_DIR, that's an
    explicit choice -- this function leaves it alone entirely (no override,
    no restore-on-exit anywhere in this module). Otherwise it points the env
    var at DEFAULT_PRED_CACHE_DIR for this run; unlike the old ephemeral
    tempdir behavior, the directory is NOT torn down afterward -- that's the
    whole point of making this persistent.

    Safety against a checkpoint retrained in place between two runs at the
    same model_path comes from kairos_strategies._model_checkpoint_fingerprint()
    being folded into the shared cache key (see kairos_strategies._shared_cache_key),
    not from tearing the directory down every run.

    Returns the resolved KAIROS_PRED_CACHE_DIR value (whichever of the two
    cases applied).
    """
    existing = os.environ.get("KAIROS_PRED_CACHE_DIR")
    if existing:
        return existing
    os.makedirs(DEFAULT_PRED_CACHE_DIR, exist_ok=True)
    os.environ["KAIROS_PRED_CACHE_DIR"] = DEFAULT_PRED_CACHE_DIR
    return DEFAULT_PRED_CACHE_DIR


def prewarm_prediction_cache(base_now, interval, months_back, run_kwargs, notify: bool = True):
    """Populate kairos_predcache's shared disk/memory cache MODEL-MAJOR
    (all dates for the base model, then all dates for each finetuned group's
    model) instead of the DATE-MAJOR order generate_and_dedupe_reports'
    day-by-day kairos_signals.run() calls use.

    Motivation: kairos_strategies._ensure_model_loaded() is a single global
    slot -- reloading a different model_path triggers a full unload+reload
    (HF from_pretrained + GPU move). run()'s date-major loop visits
    base -> finetuned-group-1 -> ... -> next date -> base again, reloading
    the model G+1 times PER DATE for G finetuned groups. Sweeping
    model-major here means the actual generate_and_dedupe_reports() call
    that follows finds every (symbol, bar, model) prediction already cached
    (see kairos_strategies.predict_all_batch's cache-before-load path), so
    its date-major loop never triggers a real reload.

    Discards predict_all_batch's return value -- the only purpose of this
    sweep is the kairos_predcache disk-cache side effect (requires
    KAIROS_PRED_CACHE_DIR to already be set in the environment; see main(),
    which points it at the persistent DEFAULT_PRED_CACHE_DIR by default).

    Each sweep unit -- the base sweep (one unit covering every group) and
    each finetuned group's sweep (its own unit, since each has a distinct
    model_path) -- runs as two passes:
      1. A fetch+cache-check pass: fetch each (group, date)'s data (each
         fetch wrapped in its own try/except, mirroring kairos_signals.run()'s
         Pass 1/Pass 2 loops, so one bad date/symbol never aborts the whole
         sweep) and check it against kairos_strategies.is_batch_cached() --
         a read-only lookup that never loads a model. The MOMENT any entry
         is a genuine miss, the check pass stops entirely (2026-07-29:
         previously it kept checking every remaining entry even after
         already knowing a load was needed -- pure wasted fetch+lookup work,
         since one miss is all the information needed to answer "does this
         unit need a real load?").
      2. If step 1 ever found a miss, send one _notify() heads-up (via
         _format_prewarm_load_message) naming the model and the sweep's
         date range/count, BEFORE the model actually loads -- then a real
         predict pass over the FULL (group, date) cross product for this
         unit (not just what the check pass happened to cover before
         stopping early -- predict_all_batch() does its own fetch + cache
         lookup per entry regardless, so the load pass never depended on
         the check pass's coverage). If the check pass completed with zero
         misses, the load pass is skipped entirely (nothing to load) and a
         line is printed to say so.

    Returns the list of (context, error) failure strings collected along
    the way (empty list if nothing failed).

    MEMORY NOTE (2026-07-29): earlier versions of this function fetched each
    (group, date)'s data ONCE in the check pass and kept every fetched
    DataFrame alive in a list (base_entries / group_entries) for the load
    pass to consume afterward. For a 6-month/1d run that's ~150 groups x
    ~183 dates of lookback-window DataFrames held simultaneously -- a live
    run was caught by a kill-at-10GB monitor: RSS climbed linearly from
    ~2.4GB to 10.1GB in 18 minutes while still inside the BASE sweep's check
    pass (data/predcache hadn't gained a single new file), matching this
    exact accumulation. That per-entry list no longer exists at all (see the
    early-exit behavior above) -- the load pass, on the rare occasion it
    runs, builds its own full cross-product freshly and re-fetches each
    entry's DataFrame right before use, so at most one entry's data is
    resident at a time. fetch_data_raw reads from the
    local price_cache (SQLite, remote=False) -- re-fetching doubles that
    local I/O but is far cheaper than holding tens of thousands of
    DataFrames in RAM for the whole sweep.
    """

    step = _interval_to_timedelta(interval)
    days_per_step = step.total_seconds() / 86400.0
    n_iterations = round(months_back * 30.44 / days_per_step)
    dates = [base_now - i * step for i in range(n_iterations)]

    db_path = run_kwargs.get("db_path", DB_PATH)
    base_only = run_kwargs.get("base_only", False)
    lookback = kairos_strategies.LOOKBACK

    # See kairos_signals._connect_with_retry's docstring for why this isn't
    # a plain sqlite3.connect() -- a live run crashed here (and in run()'s
    # date-major loop) with a transient "unable to open database file".
    conn = _kairos_signals_mod._connect_with_retry(db_path)
    try:
        rows = _kairos_signals_mod.load_work_items(conn, intervals=[interval])
        groups = _kairos_signals_mod.group_items(rows)
        accepted_finetuned = (
            {} if base_only else _kairos_signals_mod.load_accepted_finetuned(conn)
        )
    finally:
        conn.close()

    failures = []

    # ── Base sweep: one unit covering every group, every date. ─────────────
    # Check pass stops the instant a miss is found (see docstring); the load
    # pass, when it runs, always covers the full (group, date) cross product
    # directly rather than a list built by the check pass.
    base_needs_load = False
    base_check_pbar = tqdm(
        total=len(groups) * len(dates), desc="Prewarm check: Base", unit="req",
    )
    for (assets_str, grp_interval), _group_rows in groups.items():
        if base_needs_load:
            break
        assets = assets_str.split(",")
        for date in dates:
            base_check_pbar.set_postfix_str(f"{assets_str} @ {date:%Y-%m-%d}")
            try:
                data = {
                    sym: kairos_strategies.fetch_data_raw(sym, lookback, as_of=date).tail(lookback)
                    for sym in assets
                }
            except Exception as e:
                failures.append(
                    f"prewarm base group assets={assets_str} interval={grp_interval} "
                    f"date={date}: {e}"
                )
                base_check_pbar.update(1)
                continue
            base_check_pbar.update(1)
            if not kairos_strategies.is_batch_cached(data, model_path=None):
                base_needs_load = True
                break
            if base_check_pbar.n % _PREWARM_GC_INTERVAL == 0:
                gc.collect()
    base_check_pbar.close()

    if base_needs_load and dates:
        _notify(_format_prewarm_load_message("Base", dates), enabled=notify)
        base_all_entries = [
            (assets_str, grp_interval, date)
            for (assets_str, grp_interval) in groups.keys()
            for date in dates
        ]
        base_load_pbar = tqdm(
            base_all_entries, desc="Prewarm load: Base", unit="req", total=len(base_all_entries),
        )
        for i, (assets_str, grp_interval, date) in enumerate(base_load_pbar, start=1):
            base_load_pbar.set_postfix_str(f"{assets_str} @ {date:%Y-%m-%d}")
            assets = assets_str.split(",")
            try:
                data = {
                    sym: kairos_strategies.fetch_data_raw(sym, lookback, as_of=date).tail(lookback)
                    for sym in assets
                }
                kairos_strategies.predict_all_batch(data, model_path=None)
            except Exception as e:
                failures.append(
                    f"prewarm base group assets={assets_str} interval={grp_interval} "
                    f"date={date}: {e}"
                )
                continue
            finally:
                if i % _PREWARM_GC_INTERVAL == 0:
                    gc.collect()
    else:
        print(
            f"Prewarm load: Base skipped -- check pass found no cache misses "
            f"({len(groups) * len(dates)} entries already warm)."
        )

    # ── Finetuned sweep: each group with an accepted finetuned model is its
    # own unit (own fetch+check pass, own possible notification, own
    # predict pass) -- distinct model_path means a distinct load event. ────
    if not base_only and accepted_finetuned:
        for (assets_str, grp_interval), _group_rows in groups.items():
            sorted_key = ",".join(sorted(assets_str.split(",")))
            model_path = accepted_finetuned.get((sorted_key, grp_interval))
            if model_path is None:
                continue

            assets = assets_str.split(",")
            model_label = f"Finetuned({assets_str})"
            group_needs_load = False
            group_check_pbar = tqdm(
                dates, desc=f"Prewarm check: {model_label}", unit="req", total=len(dates),
            )
            for i, date in enumerate(group_check_pbar, start=1):
                group_check_pbar.set_postfix_str(f"{assets_str} @ {date:%Y-%m-%d}")
                try:
                    data = {
                        sym: kairos_strategies.fetch_data_raw(sym, lookback, as_of=date).tail(lookback)
                        for sym in assets
                    }
                except Exception as e:
                    failures.append(
                        f"prewarm finetuned group assets={assets_str} interval={grp_interval} "
                        f"date={date} (model_path={model_path}): {e}"
                    )
                    continue
                if not kairos_strategies.is_batch_cached(data, model_path=model_path):
                    group_needs_load = True
                    break
                if i % _PREWARM_GC_INTERVAL == 0:
                    gc.collect()

            if group_needs_load and dates:
                _notify(_format_prewarm_load_message(model_label, dates), enabled=notify)
                group_load_pbar = tqdm(
                    dates, desc=f"Prewarm load: {model_label}", unit="req", total=len(dates),
                )
                for i, date in enumerate(group_load_pbar, start=1):
                    group_load_pbar.set_postfix_str(f"{assets_str} @ {date:%Y-%m-%d}")
                    try:
                        data = {
                            sym: kairos_strategies.fetch_data_raw(sym, lookback, as_of=date).tail(lookback)
                            for sym in assets
                        }
                        kairos_strategies.predict_all_batch(data, model_path=model_path)
                    except Exception as e:
                        failures.append(
                            f"prewarm finetuned group assets={assets_str} interval={grp_interval} "
                            f"date={date} (model_path={model_path}): {e}"
                        )
                        continue
                    finally:
                        if i % _PREWARM_GC_INTERVAL == 0:
                            gc.collect()
            else:
                print(
                    f"Prewarm load: {model_label} skipped -- check pass found no "
                    f"cache misses ({len(dates)} entries already warm)."
                )

    return failures


_CFD_TICKER_RE = re.compile(r"(=F|=X|-USD)$")


def map_instrument_type(candidate_or_row):
    """Map a Candidate (or an allocation.py result-row dict) to Phantom
    Ledger's InstrumentType ("stock" or "cfd").

    "cfd" for any short direction, or a ticker matching Kairos's futures
    (e.g. "CL=F", "NG=F"), forex (e.g. "EURUSD=X", "AUDCAD=X"), or crypto
    (e.g. "BTC-USD", "WIF-USD", "UNI7083-USD") ticker conventions (see real
    examples in results/*.md); "stock" for everything else (plain equity
    tickers like "AAPL", "NFLX").
    """
    if isinstance(candidate_or_row, dict):
        ticker = candidate_or_row.get("ticker", "") or ""
        direction = candidate_or_row.get("direction", "") or ""
    else:
        ticker = getattr(candidate_or_row, "ticker", "") or ""
        direction = getattr(candidate_or_row, "direction", "") or ""

    if direction == "short":
        return "cfd"
    if ticker and _CFD_TICKER_RE.search(ticker):
        return "cfd"
    return "stock"


def _get_field(obj, key):
    """Duck-typed field access: works for dicts and attribute-bearing
    objects (e.g. phantom.models.position.Position) alike."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def compute_corrected_realized_pnl(position):
    """True per-trade economic P&L, correcting a confirmed `phantom_ledger`
    accounting bug (see docs/papertrade_loss_analysis.md, "1. Equity/PnL
    accounting & reporting" for the full derivation and reproduction).

    The stored `realized_pnl` (phantom/engine/position_manager.py,
    `PositionManager.close()`, ~lines 314-327) is direction-AWARE -- it
    already flips gross P&L's sign correctly for "short" positions
    (`gross_pnl = (entry_price - exit_price) * quantity` for shorts) -- but
    it OMITS `fx_conversion_cost` from its cost deduction (`all_costs` only
    sums commission + spread + slippage). That fx cost IS real money that
    left the account: it's charged to `account.cash` at entry via
    `phantom/engine/order_manager.py`'s `OrderManager.handle_fill`
    (~line 300: `total_deduction = order.fill_price * order.quantity +
    costs.total`, where `costs.total` includes fx), but `close()` never
    subtracts it back out of `realized_pnl`. This helper applies that one
    missing correction. (There is a SEPARATE, larger phantom bug in how
    `account.cash` itself is tracked for short positions -- see
    `build_closed_trade_equity_curve`'s docstring -- but it does not affect
    `realized_pnl`'s own direction, only phantom's raw cash/equity curve.)

    Duck-typed (dict or attribute object, matching `compute_pct_profit_per_trade`).
    Returns None if realized_pnl is unavailable.
    """
    realized_pnl = _get_field(position, "realized_pnl")
    if realized_pnl is None:
        return None
    fx_cost = _get_field(position, "fx_conversion_cost") or 0.0
    return realized_pnl - fx_cost


def compute_pct_profit_per_trade(closed_positions):
    """Mean of corrected_realized_pnl / (entry_price * quantity) across
    closed positions, as a percentage (see compute_corrected_realized_pnl
    for the fx-omission correction applied to realized_pnl). Accepts dicts
    or objects with realized_pnl/entry_price/quantity attributes (no
    `phantom` import required -- pure math over duck-typed inputs).

    Returns None if there are no positions with computable P&L.
    """
    pcts = []
    for pos in closed_positions:
        realized_pnl = compute_corrected_realized_pnl(pos)
        entry_price = _get_field(pos, "entry_price")
        quantity = _get_field(pos, "quantity")
        if realized_pnl is None or not entry_price or not quantity:
            continue
        denom = entry_price * quantity
        if not denom:
            continue
        pcts.append(realized_pnl / denom * 100.0)
    if not pcts:
        return None
    return sum(pcts) / len(pcts)


def write_json_report(metrics: dict, meta: dict, out_path) -> str:
    """Write {**metrics, "meta": meta} as JSON to out_path. Returns out_path
    (as str)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    payload["meta"] = meta
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return str(out_path)


def _naive(dt):
    """Strip tzinfo for cross-source datetime comparisons (we only care
    about relative ordering here, not absolute zone)."""
    if dt is not None and getattr(dt, "tzinfo", None) is not None:
        return dt.replace(tzinfo=None)
    return dt


def _parse_iso(value):
    if isinstance(value, datetime):
        return _naive(value)
    return _naive(datetime.fromisoformat(value))


# =============================================================================
# Phantom Ledger orchestration (requires a live `phantom` install)
# =============================================================================

def floor_dt(dt: datetime, interval: timedelta) -> datetime:
    ref = datetime.min
    delta = dt - ref
    return ref + (delta // interval) * interval

def selected_rows(allocation_result):
    """Rows from an AllocationResult with status == 'SELECTED' (alloc > 0)."""
    return [row for row in allocation_result.rows if row.get("status") == "SELECTED"]


def _ensure_broker_profile(client, broker_name):
    """Load `broker_name`'s profile from phantom_ledger's own bundled
    profiles/ directory into this Phantom instance's DB, if not already
    loaded there.

    A fresh Phantom(data_dir=...) DB ships with NO broker profiles seeded
    (verified from source: BrokerRepo starts empty; phantom_ledger's own CLI
    requires an explicit one-time `phantom broker load <path>` per DB). This
    mirrors that: idempotent, safe to call every run.
    """
    from phantom.errors import NotFoundError as PhNotFoundError

    try:
        client.brokers.get(broker_name)
        return
    except PhNotFoundError:
        pass

    import phantom.profiles as _profiles_pkg
    profile_path = os.path.join(
        os.path.dirname(_profiles_pkg.__file__), f"{broker_name.lower()}.json"
    )
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"No bundled Phantom Ledger broker profile for {broker_name!r} "
            f"at {profile_path}; load one manually via client.brokers.load(...)."
        )
    client.brokers.load(profile_path)


def remove_all_open_positions(ph_instance, account_id, account_name):
    """Remove every position still open when the replay window ends, rather
    than manufacturing a same-day "manual" close at the last available price.

    A position still open at window-end never reached a genuine,
    strategy-intended conclusion (its stop-loss/take-profit hadn't actually
    resolved within the window) -- force-closing it would inject an
    arbitrary same-day exit outcome into the trade statistics that the
    strategy never actually produced. Instead this REMOVES it entirely:
    refunds its entry-side cash impact and deletes the row, so it counts as
    neither a win, a loss, nor a trade -- as if it had never been opened.

    Design decisions:
    - DELETE, not a new `status` value: phantom_ledger's `positions.status`
      column CHECK constraint only allows 'open'|'closed'|'liquidated'
      (verified via `.schema positions` against the frozen fixture DB) --
      there is no "cancelled"/"removed" value to use instead, and
      'liquidated' has forced-margin-liquidation connotations that don't
      apply here (this is "the window ended", not "the broker force-closed
      you on a margin call"). No public PositionAPI method covers this
      either (only close/get/list/modify/reset_replay), so this reaches
      directly into `ph_instance._conn` -- the same pattern
      kairos_papertrade.py already uses elsewhere (constructing
      `phantom.models.equity_point.EquityPoint` directly) when the public
      API doesn't cover an exact need.
    - Refund scope: EXACTLY what phantom's own `OrderManager.handle_fill`
      deducted from cash at entry -- `entry_price * quantity` plus the four
      entry-side cost fields it persisted onto the position row
      (commission_entry, spread_cost, slippage_cost, fx_conversion_cost).
      Verified against phantom's actual source
      (phantom/engine/order_manager.py, `handle_fill`, ~line 300:
      `total_deduction = order.fill_price * order.quantity + costs.total`,
      and ~lines 320-323, which persist each component of that SAME `costs`
      object onto the new Position unchanged) -- these four stored fields
      are exactly the entry-side costs charged, NOT a round-trip total (the
      exit-side commission/spread/slippage are computed fresh at close time
      in `position_manager.py`'s `close()` and are never written back onto
      these same columns). Refunding entry_price*quantity + these four
      fields exactly reverses the entry deduction, leaving cash as if the
      position had never been opened.
    - No AccountAPI method adjusts cash directly either (only
      create/delete/get/get_aggregate_equity/get_margin_summary), so cash is
      updated via a direct UPDATE on the same `accounts` row phantom's own
      AccountRepo would touch.
    - FK note: phantom_ledger runs with `PRAGMA foreign_keys = ON`
      (phantom/db/database.py) and `orders.position_id REFERENCES
      positions(id)` has no ON DELETE clause (defaults to RESTRICT), so
      deleting a position whose entry order still points at it would raise
      sqlite3.IntegrityError. Null out that FK first -- the order row itself
      (showing the entry actually filled) is left intact, only its now-dangling
      link to the removed position is cleared.
    """
    open_positions = ph_instance.positions.list(account_name=account_name, status="open")
    if not open_positions:
        return

    refund_total = sum(
        pos.entry_price * pos.quantity
        + pos.commission_entry + pos.spread_cost
        + pos.slippage_cost + pos.fx_conversion_cost
        for pos in open_positions
    )

    conn = ph_instance._conn
    cur = conn.cursor()
    ids = [(pos.id,) for pos in open_positions]
    cur.executemany("UPDATE orders SET position_id = NULL WHERE position_id = ?", ids)
    cur.executemany("DELETE FROM positions WHERE id = ?", ids)
    cur.execute("UPDATE accounts SET cash = cash + ? WHERE id = ?", (refund_total, account_id))
    conn.commit()


def build_closed_trade_equity_curve(closed_positions, capital, start_dt=None):
    """Build a step-function equity curve from CLOSED positions only, using
    compute_corrected_realized_pnl (direction + fx corrected), sorted by
    exit_datetime -- one point per trade close, prefixed with a starting
    point at `capital`. Returns a list of `phantom.reports.metrics.EquityPoint`.

    WHY NOT phantom's own per-bar `accounts.get_aggregate_equity()` curve:
    that curve is built from phantom's own bar-by-bar `account.cash`
    tracking, which has a CONFIRMED direction-blind bug for "short"
    positions. Root cause (see docs/papertrade_loss_analysis.md, "1.
    Equity/PnL accounting & reporting" for the full reproduction against
    the frozen fixture DB): both the entry-side cash debit
    (phantom/engine/order_manager.py, `OrderManager.handle_fill`, ~line 300:
    `total_deduction = order.fill_price * order.quantity + costs.total`)
    and every exit-side cash credit (phantom/engine/simulation_engine.py,
    `SimulationEngine.run_backtest`, ~line 214:
    `cash_return = exit_price * position.quantity - exit_costs.total`; and
    phantom/api/positions.py, `PositionAPI.close`, ~lines 93 and 128, same
    pattern) use RAW `price * quantity` with no `position.direction` check
    at all. That's correct for "long" positions (cash effect nets to
    `(exit-entry)*quantity - costs`, matching gross P&L's sign), but for
    "short" positions it's backwards: cash still moves by
    `(exit-entry)*quantity`, the OPPOSITE sign of a short's real gross P&L
    of `(entry-exit)*quantity`, so a WINNING short trade DECREASES
    phantom's tracked cash and a LOSING short INCREASES it -- even though
    `realized_pnl` itself (phantom/engine/position_manager.py,
    `PositionManager.close`, ~lines 314-317) correctly computes
    direction-aware gross P&L. Verified by exact reconciliation against the
    frozen fixture DB (`tests/data/kairos_papertrade_20260723_phantom.db`):
    `capital + sum(actual per-position cash effect, using phantom's own
    direction-blind formula)` reproduces the account's real final cash to
    12 significant figures, and the gap between that and the naive
    `capital + sum(realized_pnl)` reconciliation is EXACTLY
    `2 * sum(gross_pnl over short positions) + sum(fx_conversion_cost)`
    (~€48.71 on that run: ~€39.05 from the short-direction bug + ~€9.67 from
    the fx omission compute_corrected_realized_pnl already fixes).

    Tradeoff of this curve vs. phantom's: this is a "closed-trade" equity
    curve (a step function at each trade's exit), not a true continuous
    mark-to-market series -- it does not capture intra-trade unrealized
    drawdown from positions that are still open at some intermediate point.
    Given phantom's own continuous series can't be trusted whenever shorts
    are present, this is the most honest approximation available from data
    phantom exposes correctly.

    `start_dt` anchors the initial (capital) point; if omitted, falls back
    to the earliest closed position's entry_datetime, or `datetime.now()`
    if there are no closed positions at all.
    """
    from phantom.reports.metrics import EquityPoint as MetricsEquityPoint

    dated_pnls = []
    entry_dts = []
    for pos in closed_positions:
        exit_dt = _get_field(pos, "exit_datetime")
        entry_dt = _get_field(pos, "entry_datetime")
        if entry_dt is not None:
            entry_dts.append(_naive(entry_dt))
        pnl = compute_corrected_realized_pnl(pos)
        if exit_dt is None or pnl is None:
            continue
        dated_pnls.append((_naive(exit_dt), pnl))
    dated_pnls.sort(key=lambda t: t[0])

    if start_dt is not None:
        first_ts = _naive(start_dt)
    elif entry_dts:
        first_ts = min(entry_dts)
    elif dated_pnls:
        first_ts = dated_pnls[0][0]
    else:
        first_ts = datetime.now()

    equity = capital
    curve = [MetricsEquityPoint(timestamp=first_ts, equity=equity)]
    for ts, pnl in dated_pnls:
        equity += pnl
        curve.append(MetricsEquityPoint(timestamp=ts, equity=equity))
    return curve


def _reconcile_cash_and_log(ph_instance, account_id, capital, closed_positions, total_profit_eur):
    """Compare Kairos's own corrected total P&L against phantom's raw
    `account.cash` and log a warning (does not raise) if they diverge
    beyond a small tolerance.

    This is EXPECTED to fire whenever the run holds any "short" positions
    -- see build_closed_trade_equity_curve's docstring for the confirmed
    phantom_ledger bug that causes it (direction-blind cash debit/credit).
    Its job is to make a future divergence visible immediately (a live run,
    a long-only run where it should NOT fire, or an eventual upstream
    phantom fix) instead of requiring manual SQL forensics again, the way
    this exact gap was originally found.
    """
    try:
        raw_cash = ph_instance.accounts.get(account_id).cash
    except Exception as e:  # pragma: no cover - defensive, metrics must not hard-fail on this
        print(f"WARNING: cash reconciliation check itself failed: {e}", file=sys.stderr)
        return None

    corrected_final = capital + total_profit_eur
    gap = raw_cash - corrected_final
    if abs(gap) > 0.01:
        n_short = sum(1 for p in closed_positions if _get_field(p, "direction") == "short")
        print(
            f"WARNING: cash reconciliation gap of {gap:.4f} EUR between phantom's raw "
            f"account.cash ({raw_cash:.4f}) and Kairos's corrected total "
            f"(capital + corrected P&L = {corrected_final:.4f}) across "
            f"{len(closed_positions)} closed positions ({n_short} short). This is EXPECTED "
            f"whenever short positions are present (see docs/papertrade_loss_analysis.md, "
            f"\"1. Equity/PnL accounting & reporting\", for the confirmed phantom_ledger "
            f"root cause); if n_short == 0 and this still fires, treat it as a NEW, "
            f"uninvestigated divergence.",
            file=sys.stderr,
        )
    return gap


def compute_final_metrics(ph_instance, account_id, account_name, capital, start_dt=None) -> dict:
    """Compute the 6 required summary metrics for the finished papertrade run.

    Uses a Kairos-reconstructed "closed-trade" equity curve
    (build_closed_trade_equity_curve) rather than phantom's own
    accounts.get_aggregate_equity() -- see that function's docstring for
    the confirmed phantom_ledger direction-blind cash bug that makes
    phantom's own per-bar curve untrustworthy whenever short positions are
    involved.
    """
    from phantom.reports.metrics import calculate_metrics
    from phantom.errors import ValidationError as PhValidationError

    closed_positions = ph_instance.positions.list(account_name=account_name, status="closed")
    equity_curve = build_closed_trade_equity_curve(closed_positions, capital, start_dt=start_dt)

    equity_metrics = None
    if len(equity_curve) >= 2:
        try:
            equity_metrics = calculate_metrics(equity_curve)
        except PhValidationError:
            equity_metrics = None

    final_equity = equity_curve[-1].equity if equity_curve else capital
    total_profit_eur = final_equity - capital
    pct_profit = (
        equity_metrics.total_return_pct if equity_metrics is not None
        else (total_profit_eur / capital * 100.0 if capital else 0.0)
    )

    metrics = {
        "total_profit_eur": total_profit_eur,
        "pct_profit": pct_profit,
        "pct_profit_per_trade": compute_pct_profit_per_trade(closed_positions),
        "pct_max_drawdown": equity_metrics.max_drawdown_pct if equity_metrics is not None else 0.0,
        "sharpe": equity_metrics.sharpe_ratio if equity_metrics is not None else 0.0,
        "num_trades": len(closed_positions),
    }

    _reconcile_cash_and_log(ph_instance, account_id, capital, closed_positions, total_profit_eur)

    return metrics


def write_html_report(equity_curve, positions, metrics, meta, out_path) -> str:
    """Render the equity/cash curve + per-position markers + metrics table
    as an interactive Plotly HTML report, following the make_subplots /
    go.Table / fig.write_html(..., include_plotlyjs='cdn') idiom from
    examples/run_backtest_kairos_html.py::plot_results_html."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [_parse_iso(pt.timestamp) for pt in equity_curve]
    equity_vals = [pt.equity for pt in equity_curve]
    cash_vals = [getattr(pt, "cash", pt.equity) for pt in equity_curve]

    def _equity_near(dt):
        dt = _naive(dt)
        best = None
        for x, y in zip(xs, equity_vals):
            if x <= dt and (best is None or x > best[0]):
                best = (x, y)
        if best is not None:
            return best[1]
        return equity_vals[0] if equity_vals else 0.0

    fig = make_subplots(
        rows=2, cols=1, row_heights=[4.0, 1.4],
        specs=[[{"type": "xy"}], [{"type": "table"}]],
        vertical_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=xs, y=equity_vals, name="Equity (total)",
        line=dict(color="#42a5f5", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=xs, y=cash_vals, name="Cash (available)",
        line=dict(color="#2ecc71", width=2),
    ), row=1, col=1)

    for pos in positions:
        entry_dt = _naive(pos.entry_datetime)
        exit_dt = _naive(pos.exit_datetime) if pos.exit_datetime is not None else entry_dt
        y0 = _equity_near(entry_dt)
        y1 = _equity_near(exit_dt)
        exit_price_str = f"{pos.exit_price:.4f}" if pos.exit_price is not None else "n/a"
        pnl_str = f"{pos.realized_pnl:.2f}" if pos.realized_pnl is not None else "n/a"
        hover = (
            f"{pos.ticker} ({pos.direction})<br>"
            f"Entry: {pos.entry_price:.4f} @ {entry_dt}<br>"
            f"Exit: {exit_price_str} @ {exit_dt}<br>"
            f"PnL: {pnl_str}<extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=[entry_dt, exit_dt], y=[y0, y1],
            mode="lines+markers",
            line=dict(color="gray", dash="dot", width=1),
            marker=dict(color="red", size=8, symbol="circle"),
            name=pos.ticker, showlegend=False,
            hovertemplate=hover,
        ), row=1, col=1)

    metric_labels = list(metrics.keys())
    metric_values = [
        f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()
    ]
    fig.add_trace(go.Table(
        header=dict(values=["Metric", "Value"], fill_color="#1e293b",
                    font=dict(color="white", size=12)),
        cells=dict(values=[metric_labels, metric_values], fill_color="#0f172a",
                   font=dict(color="#94a3b8", size=11, family="monospace")),
    ), row=2, col=1)

    title = out_path.name
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=14)),
        height=900, showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
    )
    fig.update_yaxes(row=1, col=1, title_text="Equity")

    fig.write_html(str(out_path), include_plotlyjs="cdn")

    paragraph = (
        "<p style=\"font-family:sans-serif;color:#cbd5e1;background:#0f172a;"
        "padding:10px 16px;margin:0;\">"
        f"Paper-trade backtest of Kairos signals from {meta.get('start')} to "
        f"{meta.get('end')} ({meta.get('interval')} bars, "
        f"{meta.get('months_back')} months back), starting capital "
        f"{meta.get('capital')} {meta.get('currency', 'EUR')} on broker "
        f"{meta.get('broker')}, "
        f"{'base-model only' if meta.get('base_only') else 'including finetuned overlay'}. "
        f"{meta.get('num_days', len(equity_curve))} trading days replayed."
        "</p>"
    )
    with open(out_path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("<body>", "<body>" + paragraph, 1)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(out_path)


def _report_filename(end_dt, start_dt, interval, months_back, ext):
    return (
        f"kairos_signals_papertrade_{end_dt:%Y%m%d%H%M}_{start_dt:%Y%m%d%H%M}_"
        f"{interval}_{months_back}m.{ext}"
    )


def _format_start_message(base_now, args) -> str:
    """🟢 start-of-run notification text, sent right before the expensive
    (potentially multi-hour) generate_and_dedupe_reports() call."""
    return (
        f"🟢 Kairos papertrade starting: window ending {base_now.strftime('%Y-%m-%d %H:%M')}, "
        f"interval={args.interval}, months_back={args.months_back}, top_n={args.top_n}, "
        f"capital={args.capital}, broker={args.broker}, base_only={args.base_only}"
    )

def _format_start_sim_message(base_now, args) -> str:
    return (
        f"🟢 Kairos papertrade simulating: window ending {base_now.strftime('%Y-%m-%d %H:%M')}, "
        f"interval={args.interval}, months_back={args.months_back}, top_n={args.top_n}, "
        f"capital={args.capital}, broker={args.broker}, base_only={args.base_only}"
    )

def _format_finish_message(metrics: dict, report_filename: str) -> str:
    """✅ success notification text, summarizing the final metrics dict
    returned by compute_final_metrics()."""
    return (
        f"✅ Kairos papertrade finished: total_profit_eur="
        f"{metrics.get('total_profit_eur', 0.0):.2f}, pct_profit="
        f"{metrics.get('pct_profit', 0.0):.2f}%, sharpe={metrics.get('sharpe', 0.0):.2f}, "
        f"num_trades={metrics.get('num_trades', 0)}. Report: {report_filename}"
    )


def _format_crash_message(exc: Exception, base_now, args) -> str:
    """💥 unhandled-crash notification text, with a traceback tail."""
    tb_tail = traceback.format_exc()[-2000:]
    return (
        f"💥 Kairos papertrade CRASHED ({type(exc).__name__}): window ending "
        f"{base_now.strftime('%Y-%m-%d %H:%M')}, interval={args.interval}, "
        f"months_back={args.months_back}\n```\n{tb_tail}\n```"
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for main()."""
    parser = argparse.ArgumentParser(
        description="Paper-trade Kairos signals through Phantom Ledger (roadmap Phase 4.1)"
    )
    parser.add_argument("--db", default=DB_PATH, help="Path to pipeline_results.db")
    parser.add_argument("--out", default=RESULTS_DIR, help="Output dir for the final JSON/HTML reports")
    parser.add_argument("--months-back", dest="months_back", type=float, default=6.0)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--top-n", dest="top_n", type=int, default=3)
    parser.add_argument("--signal-selection", dest="signal_selection", default=None,
                        help="Optional rule string that fully replaces the default "
                             "Portfolio Allocation gating (min_n/positive-EV-net) and "
                             "ranking (score sort, top_k) used both for the generated "
                             "reports and for the actual orders Phantom Ledger places. "
                             "Grammar: comma-separated clauses, each either a condition "
                             "\"'col' OP value\" (OP one of > >= < <= == !=), an "
                             "\"ORDER 'col' [ASC|DESC]\" clause (default DESC, at most one), "
                             "or a \"TOP <int>\" clause (at most one). Column names match "
                             "the Allocation sheet headers (Ticker, Cluster, Strategy, Dir, "
                             "Entry, Stop, Target, Risk %%, Reward %%, b, n, Win raw, Win "
                             "shrunk, EV raw %%, EV net %%, Kelly raw, Score, Sharpe), "
                             "case-insensitive. Example: \"'n' > 60, 'Win raw' > 0.6, "
                             "ORDER 'EV raw %%' DESC, TOP 3\". NOTE: when set, this REPLACES "
                             "(not adds to) the default min_n/EV-positivity gate -- a rule "
                             "that doesn't check EV can admit a negative-EV signal. "
                             "--top-n remains the fallback selection size when the rule "
                             "has no TOP clause of its own.")
    parser.add_argument("--capital", type=float, default=200.0)
    parser.add_argument("--broker", default="IBKR")
    parser.add_argument("--base-only", dest="base_only", action="store_true", default=False,
                        help="Skip the accepted-finetuned overlay pass and use only the base model (default: off — finetuned overlay is used when an accepted model exists)")
    parser.add_argument("--include-finetuned", dest="base_only", action="store_false",
                        help="Include the accepted-finetuned overlay pass (default: on)")
    # Matches the measured realized cost per docs/papertrade_loss_analysis.md Factor 7
    parser.add_argument("--min_ev_pct", type=float, default=0.15)
    parser.add_argument("--cluster_map", default=None)
    parser.add_argument("--phantom-data-dir", dest="phantom_data_dir",
                        default=DEFAULT_PHANTOM_DATA_DIR)
    parser.add_argument("--html", action="store_true", default=False)
    parser.add_argument("--effective_per", default=None,
                        help='Override "now" (end of window): \'YYYYMMDD [HHnn]\'')
    parser.add_argument("--account-name", dest="account_name", default=None)
    parser.add_argument("--no-telegram", dest="notify", action="store_false",
                         default=True,
                         help="Disable Telegram notifications for this run (default: "
                              "notifications enabled, sent via kairos.ops.send_telegram using "
                              "TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID from the environment - see "
                              ".env.example)")
    parser.add_argument("--no-pred-cache", dest="pred_cache", action="store_false",
                         default=True,
                         help="Disable the shared kairos_predcache prewarm/reuse pass for this "
                              "run (default: enabled -- a persistent KAIROS_PRED_CACHE_DIR under "
                              "data/predcache/ is used (unless already set in the environment), "
                              "model-major prewarm_prediction_cache() populates it, and "
                              "generate_and_dedupe_reports() reuses it so kairos_strategies' "
                              "single-slot model loader isn't reloaded once per (date, group); "
                              "the cache survives across invocations and is safe against "
                              "in-place model retraining -- see DEFAULT_PRED_CACHE_DIR)")
    return parser


def _raise_fd_limit() -> None:
    """Raise this process's own open-file soft limit to its hard limit.

    A live 6-month run crashed with `OSError: [Errno 24] Too many open
    files` (2026-07-29) writing a report -- not a memory issue, a real file
    descriptor exhaustion, hours into a run that fetches price data for
    ~150 groups x 183 dates x several symbols each. Whatever the parent
    shell/session handed this process as its starting soft RLIMIT_NOFILE
    (which can differ from what an interactive `ulimit -n` check shows --
    login shells, cron, systemd, and `uv run` can all start a process with
    a different default), the hard limit is usually far higher and this
    process is always allowed to raise its own soft limit up to it without
    elevated privileges. Best-effort: some sandboxed/containerized
    environments forbid even that, so a failure here is swallowed rather
    than crashing before the real work starts.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if hard != resource.RLIM_INFINITY and soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"[kairos_papertrade] Raised open-file limit: {soft} -> {hard}", file=sys.stderr)
    except (ValueError, OSError) as exc:
        print(f"[kairos_papertrade] WARNING: could not raise open-file limit: {exc}", file=sys.stderr)

def main(argv=None):
    _raise_fd_limit()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

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
    base_now = now if now is not None else datetime.now()

    _notify(_format_start_message(base_now, args), enabled=args.notify)

    try:
        # PHANTOM_DATA must be set BEFORE `import phantom` so its DB/price-cache
        # lookups land in an isolated directory, not Kairos's own data/ tree.
        os.makedirs(args.phantom_data_dir, exist_ok=True)
        os.environ["PHANTOM_DATA"] = args.phantom_data_dir

        import phantom as ph
        from phantom.models.order import Order
        from allocation import fetch_signals, allocate, AllocationConfig, load_cluster_map

        run_kwargs = dict(
            db_path=args.db, out_dir=args.out,
            min_ev_pct=args.min_ev_pct,
            cluster_map_path=args.cluster_map,
            base_only=args.base_only,
            signal_selection=parsed_signal_selection,
        )

        # Hold the SHARED kairos.ops.GpuLock (the same one daily_signals,
        # weekly_discovery, and finetune_next all respect) across the whole
        # model-inference loop, not just this function's own bookkeeping --
        # a papertrade run can take hours, and without this lock
        # finetune_next's is_gpu_idle() preflight (which only samples
        # nvidia-smi *utilization*, not VRAM) can see this process sitting
        # idle between calls and barge in, colliding on the GPU's limited
        # VRAM (observed in production: a concurrent finetune_next crashed
        # with torch.OutOfMemoryError while papertrade was running). This
        # necessarily means finetune_next/daily_signals/weekly_discovery
        # will block (and, if the lock isn't freed within GpuLock's 5-minute
        # timeout, fail with OpsError) for as long as this loop runs --
        # accepted tradeoff over the alternative of colliding outright.
        with GpuLock():
            if args.pred_cache:
                # Populate the persistent shared prediction cache model-major
                # via prewarm_prediction_cache() BEFORE the date-major
                # generate_and_dedupe_reports() loop runs, so every (symbol,
                # bar, model) prediction it needs is already cached and
                # kairos_strategies never reloads the model mid-loop. See
                # _ensure_pred_cache_dir_env() for the persistent-dir choice.
                _ensure_pred_cache_dir_env()
                prewarm_failures = prewarm_prediction_cache(
                    base_now, args.interval, args.months_back, run_kwargs, notify=args.notify,
                )
                if prewarm_failures:
                    # Previously silently discarded -- a whole finetuned
                    # group's fetch failing on every date (leaving its load
                    # pass a silent "0req" no-op) was invisible without this.
                    print(
                        f"WARNING: prewarm_prediction_cache had "
                        f"{len(prewarm_failures)} failure(s):", file=sys.stderr,
                    )
                    for msg in prewarm_failures[:20]:
                        print(f"  {msg}", file=sys.stderr)
                    if len(prewarm_failures) > 20:
                        print(f"  ... and {len(prewarm_failures) - 20} more", file=sys.stderr)
                dated_rows = generate_and_dedupe_reports(
                    base_now, args.interval, args.months_back, run_kwargs, notify=args.notify,
                )
            else:
                dated_rows = generate_and_dedupe_reports(
                    base_now, args.interval, args.months_back, run_kwargs, notify=args.notify,
                )
        if not dated_rows:
            raise RuntimeError("No kairos_signals reports were generated in the requested window.")

        _notify(_format_start_sim_message(base_now, args), enabled=args.notify)
        
        cluster_map = load_cluster_map(args.cluster_map) if args.cluster_map else {}

        client = ph.Phantom(data_dir=args.phantom_data_dir)
        intraday_provider = _IntradayFallbackProvider(args.phantom_data_dir)
        _ensure_broker_profile(client, args.broker)
        account_name = args.account_name or f"kairos_papertrade_{base_now.strftime('%Y%m%d%H%M')}"
        account = client.accounts.create(
            name=account_name, account_type="algorithm", broker=args.broker,
            capital=args.capital, currency="EUR", algorithm_id="kairos_papertrade",
            algorithm_version=args.interval,
        )
        account_id = account.id

        equity_curve = []
        prev_candidates = None
        for effective_dt, stats_rows, advice_rows in dated_rows:
            if prev_candidates:
                open_positions = client.positions.list(account_name=account_name, status="open")
                open_tickers = {p.ticker for p in open_positions}
                cash = client.accounts.get(account_id).cash
                alloc_config = AllocationConfig(
                    top_k=args.top_n, gross_cap_pct=100, equity=cash, cluster_map=cluster_map,
                    selection_rule=parsed_signal_selection,
                )
                enabled_mask = {c.ticker: (c.ticker not in open_tickers) for c in prev_candidates}
                alloc_result = allocate(prev_candidates, alloc_config, enabled_mask=enabled_mask)

                for row in selected_rows(alloc_result):
                    entry = row.get("entry")
                    if not entry:
                        continue
                    alloc_eur = row["alloc"] / 100.0 * cash
                    quantity = alloc_eur / entry
                    if quantity <= 0:
                        continue
                    order = Order(
                        account_id=account_id, ticker=row["ticker"],
                        instrument_type=map_instrument_type(row),
                        direction=row["direction"], order_type="market",
                        quantity=quantity, take_profit=row.get("target"),
                        stop_loss=row.get("stop"), created_at=effective_dt,
                    )
                    client.orders.place(account_id, order)

            all_open_tickers = {p.ticker for p in client.positions.list(account_name=account_name, status="open")}
            new_tickers = {c.ticker for c in (prev_candidates or [])}
            tickers = sorted(all_open_tickers | new_tickers)
            if tickers:
                # end must be the START OF THE NEXT DAY, not the same midnight,
                # or the daily bar (timestamped ~04-05h UTC) gets filtered out
                # by HistoricalProvider's `df.index <= end_ts` check and
                # nothing fills/evaluates.
                day_start = datetime(effective_dt.year, effective_dt.month, effective_dt.day)
                day_end = day_start + timedelta(days=1)
                backtest_start_t = time.monotonic()
                try:
                    result = client.runner.backtest(
                        account_id=account_id, tickers=tickers, start=day_start, end=day_end,
                        data_provider=intraday_provider,
                    )
                    equity_curve = result.equity_curve
                except Exception as e:
                    print(
                        f"WARNING: runner.backtest failed for {effective_dt} "
                        f"(tickers={tickers}): {e}", file=sys.stderr,
                    )
                finally:
                    backtest_elapsed = time.monotonic() - backtest_start_t
                    if backtest_elapsed > _SLOW_ITERATION_THRESHOLD_SECONDS:
                        _notify(
                            f"⏱️ Kairos papertrade: backtest for {effective_dt:%Y-%m-%d} "
                            f"took {backtest_elapsed / 60:.1f}min (>5min) — still running",
                            enabled=args.notify,
                        )
                        _log_watchdog_snapshot(
                            f"backtest {effective_dt:%Y-%m-%d}", backtest_elapsed,
                        )

            prev_candidates = fetch_signals(stats_rows, advice_rows)

        last_effective_dt = dated_rows[-1][0]
        start_dt, end_dt = dated_rows[0][0], last_effective_dt
        remove_all_open_positions(client, account_id, account_name)

        # Reflect the window-end removal in our in-memory equity_curve for the
        # HTML chart (no public API exposes a raw EquityPoint re-query;
        # reconstruct in memory from the account's post-removal cash). Unlike
        # the old force-close behavior, there's no "closed at price X" story
        # here -- removed positions are refunded and simply excluded -- so this
        # just appends the final actual cash as the chart's closing point.
        final_cash = client.accounts.get(account_id).cash
        if equity_curve:
            from phantom.models.equity_point import EquityPoint as ModelEquityPoint
            final_ts = last_effective_dt
            if final_ts.tzinfo is None:
                final_ts = final_ts.replace(tzinfo=timezone.utc)
            equity_curve = list(equity_curve) + [
                ModelEquityPoint(
                    account_id=account_id,
                    timestamp=final_ts.isoformat(),
                    equity=final_cash, cash=final_cash, unrealized_pnl=0.0,
                )
            ]

        metrics = compute_final_metrics(
            client, account_id, account_name, args.capital, start_dt=start_dt,
        )
        closed_positions = client.positions.list(account_name=account_name, status="closed")

        meta = {
            "account_name": account_name,
            "start": start_dt.isoformat(), "end": end_dt.isoformat(),
            "interval": args.interval, "months_back": args.months_back,
            "capital": args.capital, "currency": "EUR", "broker": args.broker,
            "base_only": args.base_only, "top_n": args.top_n,
            "num_days": len(dated_rows),
        }

        os.makedirs(args.out, exist_ok=True)
        json_path = os.path.join(args.out, _report_filename(end_dt, start_dt, args.interval, args.months_back, "json"))
        write_json_report(metrics, meta, json_path)
        print(json_path)

        if args.html:
            html_path = os.path.join(args.out, _report_filename(end_dt, start_dt, args.interval, args.months_back, "html"))
            write_html_report(equity_curve, closed_positions, metrics, meta, html_path)
            print(html_path)

        _notify(
            _format_finish_message(metrics, os.path.basename(json_path)),
            enabled=args.notify,
        )

        return metrics
    except Exception as exc:
        _notify(_format_crash_message(exc, base_now, args), enabled=args.notify)
        raise


if __name__ == "__main__":
    main()
