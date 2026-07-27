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
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import price_cache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kairos_signals import DB_PATH, RESULTS_DIR, _interval_to_timedelta
import kairos_signals as _kairos_signals_mod
from kairos.ops import GpuLock, OpsError, send_telegram

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PHANTOM_DATA_DIR = os.path.join(REPO_ROOT, "data", "phantom_ledger")

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
_SLOW_ITERATION_THRESHOLD_SECONDS = 300.0

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
    if not enabled:
        return
    try:
        send_telegram(text, parse_mode=None)
    except OpsError as exc:
        print(f"[kairos_papertrade] WARNING: Telegram notification failed: {exc}", file=sys.stderr)


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
    spamming a message per iteration -- only outliers notify.

    Returns a list of (effective_dt, stats_rows, advice_rows) tuples, sorted
    oldest-first.
    """
    step = _interval_to_timedelta(interval)
    days_per_step = step.total_seconds() / 86400.0
    n_iterations = round(months_back * 30.44 / days_per_step)

    seen = {}
    for i in range(n_iterations):
        iter_now = base_now - i * step
        start_t = time.monotonic()
        out_path, stats_rows, advice_rows = _kairos_signals_mod.run(
            now=iter_now, intervals=[interval], return_rows=True, **run_kwargs
        )
        elapsed = time.monotonic() - start_t
        if elapsed > _SLOW_ITERATION_THRESHOLD_SECONDS:
            _notify(
                f"⏱️ Kairos papertrade: report {i + 1}/{n_iterations} "
                f"(date {iter_now:%Y-%m-%d}) took {elapsed / 60:.1f}min (>5min) — "
                f"still running",
                enabled=notify,
            )
        effective_dt = parse_report_effective_dt(out_path)
        if effective_dt not in seen:
            seen[effective_dt] = (effective_dt, stats_rows, advice_rows)

    return [seen[key] for key in sorted(seen.keys())]


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
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

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
            dated_rows = generate_and_dedupe_reports(
                base_now, args.interval, args.months_back, run_kwargs, notify=args.notify,
            )
        if not dated_rows:
            raise RuntimeError("No kairos_signals reports were generated in the requested window.")

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
