"""E5-S15 -- Leverage-off regression and exposure-cap verification.

Unlike its siblings (`test_kairos_papertrade_loss_repro.py`,
`test_kairos_papertrade_mtm_repro.py`), which call sub-functions like
`compute_final_metrics` directly against an already-populated frozen fixture
DB, this ticket explicitly wants a test that drives the real CLI entrypoint,
`kairos_papertrade.main([...])`. `main()` needs two things this test
environment does not have (see APPENDIX-A-standards.md -- no GPU, no
network, anywhere in this suite):

1. `generate_and_dedupe_reports()` -- normally runs Kronos model inference
   over a date window and returns `dated_rows` (a list of
   `(effective_dt, stats_rows, advice_rows)` tuples).
2. `_IntradayFallbackProvider` -- normally fetches real price bars over the
   network for order fill/SL/TP evaluation and daily MTM marks.

APPROACH TAKEN (full `main()` invocation, not a fallback): both are patched
at the `kairos_papertrade` module level (where `main()` looks them up) with
deterministic, hand-built replacements:

- `generate_and_dedupe_reports` is replaced with a function that returns a
  small, fixed `dated_rows` list built from `stats_rows`/`advice_rows` dicts
  with exactly the keys `allocation.py::fetch_signals()` (~line 222) reads
  (`strategy`, `symbol`, `direction`, `entry`, `stop`, `target`,
  `expected_value`, `base_win_rate`, `base_sharpe`; and for advice rows
  `expected_value`, `entry`, `base_win_rate`, `base_signals`, `signal`).
- `_IntradayFallbackProvider` is replaced by `_FakeBarsProvider`, a small
  class satisfying the same `.get_bars(ticker, start, end) -> pd.DataFrame`
  (Open/High/Low/Close/Volume, tz-aware DatetimeIndex) /
  `.get_current_price(ticker)` contract as the real class (see its
  docstring/source ~line 284 of kairos_papertrade.py), backed by a
  hand-picked, fully deterministic OHLC table.

Everything else in `main()` runs for REAL against these synthetic-but-fixed
inputs: `allocation.fetch_signals`/`allocate` candidate selection and
sizing, `phantom_ledger` order placement/fills/SL/TP evaluation
(`client.runner.backtest`), `kairos_margin.classify_symbol`/
`kairos_mtm.admission_check`/`compute_daily_snapshot` margin math, and
`compute_final_metrics`. `GpuLock` is exercised for real too -- reading its
source (`kairos/ops.py`) confirms it is a plain `flock()` on a lock file
with no GPU/CUDA dependency, so no patch is needed. `--no-pred-cache` skips
`prewarm_prediction_cache` (real GPU/model-inference machinery, irrelevant
once `generate_and_dedupe_reports` is patched) and `--no-telegram` makes
`_notify` a silent print-only no-op (no network).

Both scenarios use a single long-only strategy across a few synthetic
trading days on plain (non-forex/futures/crypto) tickers, with candidate
win-rate/n/reward-risk parameters chosen so the Kelly-sized allocation
saturates `AllocationConfig.max_pos_pct` (15%, default, unconfigurable via
this CLI) -- i.e. every admitted position gets exactly 15% of current cash,
regardless of the exact Kelly formula's fitted value, as long as it clears
15%. This sidesteps needing to hand-reproduce the Kelly fraction to the
last decimal while still exercising the real formula.

Cost model: IBKR's bundled broker profile (`phantom/profiles/ibkr.json`)
uses per-share commission (min $1, capped at 10% of notional -- never binds
here), a *dynamic* spread model that only varies with ATR/hour-of-day
(both `None` on every call phantom's own engine makes, so effectively a
flat 0.03% of notional), fixed 0.02% slippage, and a 0.1% fx-conversion fee
charged only on ENTRY (`OrderManager.handle_fill` passes
`fx_required=(account.base_currency != "USD")`, True for our EUR account;
`PositionManager.close()`'s `exit_costs` call never passes `fx_required`,
so it defaults False -- an existing phantom_ledger asymmetry, not something
this test fixes). These are all deterministic given price/quantity, with no
external state, so Test 1's baseline P&L is independently hand-derived
below and cross-checked against the actual pinned output.
"""
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import kairos_papertrade  # noqa: E402
from kairos_margin import classify_symbol, load_margin_config  # noqa: E402

MARGIN_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "margin_ibkr.yaml"


# =============================================================================
# Shared fakes / helpers
# =============================================================================

class _FakeBarsProvider:
    """Deterministic stand-in for `kairos_papertrade._IntradayFallbackProvider`.

    Constructed the same way (`_IntradayFallbackProvider(phantom_data_dir)`)
    but backed by a hand-built `{ticker: {date: (open, high, low, close,
    volume)}}` table instead of live price_cache/network calls. Implements
    the same interface real callers use: `main()`'s day loop calls
    `.get_bars()` both for `client.runner.backtest()`'s fill/SL/TP
    evaluation and for `_fetch_day_close_bars()`'s MTM mark (both go through
    this one instance, so a ticker/date must resolve identically either
    way -- trivially true here since both reads hit the same table).
    """

    def __init__(self, bars_by_ticker):
        self._bars = bars_by_ticker

    def __call__(self, data_dir):
        # Used as a drop-in replacement for the _IntradayFallbackProvider
        # CLASS: main() does `_IntradayFallbackProvider(args.phantom_data_dir)`,
        # so this factory instance itself must be callable and return the
        # (single, shared) provider.
        return self

    def get_bars(self, ticker, start, end) -> pd.DataFrame:
        day = start.date()
        row = self._bars.get(ticker, {}).get(day)
        if row is None:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        o, h, l, c, v = row
        idx = pd.DatetimeIndex([pd.Timestamp(start, tz="UTC")])
        return pd.DataFrame(
            {"Open": [o], "High": [h], "Low": [l], "Close": [c], "Volume": [v]}, index=idx,
        )

    def get_current_price(self, ticker):
        days = self._bars.get(ticker) or {}
        if not days:
            return None
        return self._bars[ticker][max(days)][3]  # last day's close

    def get_bid_ask(self, ticker):
        price = self.get_current_price(ticker)
        return (price, price) if price is not None else (None, None)

    def get_dividends(self, ticker, start, end):
        return []  # never called on the run_backtest() path (verified: only get_bars is)


def _stats_row(
    ticker, direction, entry, stop, target, expected_value, base_win_rate=0.9,
    sharpe=1.0, strategy="TestStrategy",
):
    """One `stats_rows` dict with exactly the keys `allocation.fetch_signals()` reads."""
    return {
        "strategy": strategy, "symbol": ticker, "direction": direction,
        "entry": entry, "stop": stop, "target": target,
        "expected_value": expected_value, "base_win_rate": base_win_rate,
        "base_sharpe": sharpe, "backtest_period": "synthetic", "model": "base", "size": 0.0,
    }


def _advice_row(entry, expected_value, base_win_rate=0.9, base_signals=1000, ticker=""):
    """One `advice_rows` dict with exactly the keys `allocation.fetch_signals()` reads."""
    return {
        "expected_value": expected_value, "entry": entry, "base_win_rate": base_win_rate,
        "base_signals": base_signals, "oracle_signals": None,
        "signal": f"synthetic long {ticker}", "model": "base",
    }


def _candidate(ticker, entry, stop, target, expected_value=None, base_win_rate=0.9):
    """(stats_row, advice_row) pair for one long candidate.

    n=1000 and base_win_rate=0.9 (both hardcoded via the helpers' defaults)
    are chosen deliberately high so that, combined with a >=3:1 reward:risk
    ratio, the Kelly-sized allocation (AllocationConfig defaults: n0=100,
    kelly_mult=0.35) comfortably exceeds max_pos_pct=15 and gets capped
    there -- e.g. for entry=100/stop=90/target=140 (b=4): shrink=1000/1100
    ~=0.909, p_shrunk~=0.864, kelly_raw~=0.830, kelly_frac~=0.29 -> 29% >
    15%, so alloc is exactly 15.0% regardless of small parameter changes.
    expected_value defaults to 5% of entry, comfortably clearing the
    ev_net>0 / n>=min_n(50) gate (ev_shrunk ~= 5*0.909 = 4.5% > 0.15% cost).

    base_win_rate defaults to 0.9 (kelly_frac~29%, saturates the 15%
    leverage-off cap) but callers testing leveraged position-size scaling
    (AllocationConfig.ticker_max_leverage, added when max_pos_pct started
    scaling per-ticker by leverage) need kelly_frac to clear the NEW,
    higher cap too -- e.g. 0.99 pushes kelly_frac to ~32.6% for the same
    entry/stop/target, comfortably saturating a 30%-leveraged cap.
    """
    ev = expected_value if expected_value is not None else entry * 0.05
    return (
        _stats_row(ticker, "LONG", entry, stop, target, ev, base_win_rate=base_win_rate),
        _advice_row(entry, ev, ticker=ticker, base_win_rate=base_win_rate),
    )


def _dated_rows(days):
    """days: list of (effective_dt, [candidate_pair, ...]) -> the
    `generate_and_dedupe_reports()` return shape, sorted oldest-first."""
    rows = []
    for effective_dt, candidates in days:
        stats_rows = [c[0] for c in candidates]
        advice_rows = [c[1] for c in candidates]
        rows.append((effective_dt, stats_rows, advice_rows))
    return sorted(rows, key=lambda r: r[0])


def _run_main(monkeypatch, tmp_path, dated_rows, bars_by_ticker, argv_extra, account_name):
    monkeypatch.setattr(kairos_papertrade, "generate_and_dedupe_reports", lambda *a, **kw: dated_rows)
    monkeypatch.setattr(kairos_papertrade, "_IntradayFallbackProvider", _FakeBarsProvider(bars_by_ticker))

    phantom_dir = tmp_path / "phantom_data"
    out_dir = tmp_path / "out"
    argv = [
        "--phantom-data-dir", str(phantom_dir),
        "--out", str(out_dir),
        "--margin-config", str(MARGIN_CONFIG_PATH),
        "--account-name", account_name,
        "--no-telegram", "--no-pred-cache",
    ] + argv_extra
    metrics = kairos_papertrade.main(argv)

    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) == 1, f"expected exactly one JSON report, found {json_files}"
    import json
    report = json.loads(json_files[0].read_text())

    import phantom as ph
    client = ph.Phantom(data_dir=str(phantom_dir))
    return metrics, report["meta"], client


# =============================================================================
# Test 1: --max-leverage 1.0 regression baseline
# =============================================================================

CAPITAL_1 = 10000.0

# Hand-derived expectation for the single deterministic scenario below (see
# docstring on test_leverage_off_matches_pinned_baseline for the full
# derivation). Two trades: TICKA long 100->140 (take-profit, win), TICKB
# long 50->45 (stop-loss, loss). Both open on 2024-01-02 (day1, filled at
# the bar's Open == the candidate's stated entry price) and close on
# 2024-01-03 (day2), so num_trades=2 and no positions remain open at
# window end.
EXPECTED_METRICS_LEVERAGE_OFF = {
    "total_profit_eur": 439.775,
    "pct_profit": 4.39775,
    "num_trades": 2,
}


def test_leverage_off_matches_pinned_baseline(monkeypatch, tmp_path):
    """`--max-leverage 1.0` reproduces the legacy (pre-margin) cash path
    bit-for-bit on a short, fully synthetic window.

    Hand derivation (independently reproducing phantom's own cost engine
    from `phantom/profiles/ibkr.json` + `phantom/engine/position_manager.py`
    /`order_manager.py`, not by calling kairos_papertrade's own functions):

    TICKA: entry=100, qty=15 (15% of 10000 capital / 100), exit=140 (tp).
      gross_pnl = (140-100)*15 = 600.00
      entry costs: commission=max(1, 15*0.005)=1.00; spread=0.0003*100*15=0.45;
        slippage=0.0002*100*15=0.30; fx=0.001*100*15=1.50 (entry-only, EUR
        account) -> entry_costs_total=3.25
      exit costs (price=140): commission=1.00; spread=0.0003*140*15=0.63;
        slippage=0.0002*140*15=0.42 (no fx on exit) -> exit_costs_total=2.05
      stored realized_pnl = gross_pnl - (both commissions+spreads+slippages)
        = 600 - (1+1+0.45+0.63+0.30+0.42) = 600 - 3.80 = 596.20
      corrected (fx-adjusted) = 596.20 - 1.50 = 594.70

    TICKB: entry=50, qty=30 (15% of 10000 / 50), exit=45 (sl).
      gross_pnl = (45-50)*30 = -150.00
      entry costs: commission=max(1, 30*0.005)=1.00; spread=0.0003*50*30=0.45;
        slippage=0.0002*50*30=0.30; fx=0.001*50*30=1.50 -> entry_costs_total=3.25
      exit costs (price=45): commission=1.00; spread=0.0003*45*30=0.405;
        slippage=0.0002*45*30=0.27 -> exit_costs_total=1.675
      stored realized_pnl = -150 - (1+1+0.45+0.405+0.30+0.27) = -150 - 3.425 = -153.425
      corrected = -153.425 - 1.50 = -154.925

    total_profit_eur = 594.70 + (-154.925) = 439.775
    pct_profit = 439.775 / 10000 * 100 = 4.39775%  (phantom's calculate_metrics
      total_return_pct is a plain (final/initial - 1)*100 over the 2-point
      closed-trade equity curve -- verified by reading
      phantom/reports/metrics.py::calculate_metrics)

    margin_rejected_count == 0 is a SOURCE-LEVEL guarantee, not a
    coincidence of this scenario: `kairos_mtm.admission_check()` returns
    True unconditionally whenever `alloc_config.max_leverage <= 1.0`
    (kairos_mtm.py lines 229-230), before it ever looks at notional/margin
    -- no order can ever be MARGIN_REJECTED in a --max-leverage 1.0 run.
    """
    day0 = datetime(2024, 1, 1)
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)

    dated_rows = _dated_rows([
        (day0, [_candidate("TICKA", 100.0, 90.0, 140.0), _candidate("TICKB", 50.0, 45.0, 65.0)]),
        (day1, []),  # no new signals; just lets day1's fills happen via prior day's candidates
        (day2, []),
    ])

    bars_by_ticker = {
        "TICKA": {
            day1.date(): (100.0, 101.0, 99.0, 100.5, 1000.0),   # fills at Open=100, no SL/TP touch
            day2.date(): (105.0, 141.0, 104.0, 138.0, 1000.0),  # High>=140 -> tp @ 140 exactly
        },
        "TICKB": {
            day1.date(): (50.0, 50.5, 49.5, 50.2, 1000.0),      # fills at Open=50, no SL/TP touch
            day2.date(): (48.0, 49.0, 44.0, 44.5, 1000.0),      # Low<=45 -> sl @ 45 exactly
        },
    }

    metrics, meta, client = _run_main(
        monkeypatch, tmp_path, dated_rows, bars_by_ticker,
        argv_extra=[
            "--capital", str(CAPITAL_1), "--top-n", "3",
            "--max-leverage", "1.0", "--margin-utilization", "0.8",
        ],
        account_name="leverage_off_regression",
    )
    try:
        for key, expected in EXPECTED_METRICS_LEVERAGE_OFF.items():
            assert metrics[key] == pytest.approx(expected, rel=1e-9), key

        assert meta["margin_rejected_count"] == 0
        # phantom_ledger E17-S02: distinct from margin_rejected_count above
        # (Kairos's own pre-admission gate) -- this is phantom itself
        # rejecting a placed order at fill time. Nothing should trigger it
        # in a fully-funded, unleveraged run.
        assert meta["phantom_fill_rejected_count"] == 0

        closed = client.positions.list(account_name="leverage_off_regression", status="closed")
        assert len(closed) == 2
        assert {p.close_reason for p in closed} == {"tp", "sl"}
    finally:
        client._conn.close()


# =============================================================================
# Test 2: --max-leverage 2.0 exposure-cap bound
# =============================================================================

CAPITAL_2 = 10000.0


def test_exposure_cap_bounds_peak_gross_notional(monkeypatch, tmp_path):
    """With leverage on and a tight margin_utilization cap, peak
    `gross_notional` observed across the run's `kairos_mtm_daily` rows must
    stay within `equity * max_leverage * margin_utilization_cap /
    min_initial_margin_pct` (the ticket's formula; `min_initial_margin_pct`
    is a FRACTION here, e.g. 0.20 for 20%, matching `admission_check`'s own
    `notional * initial_margin_pct / 100.0` unit convention).

    Scenario: 6 plain tickers (TICK1..TICK6), all classifying to
    `equity_cfd` (initial_margin_pct=20%, IBKR's unmatched-ticker default
    bucket per config/margin_ibkr.yaml) -- computed via classify_symbol
    below, not hardcoded, per the ticket's explicit instruction.
    TICK1/TICK2 open on day1 (`existing_margin_used_pct=0`, nothing open
    yet -- see kairos_papertrade.py's bootstrap DailySnapshot), each sized
    to their full Stage-1-capped 30% (max_pos_pct=15 * effective_leverage=2
    for --max-leverage 2.0), establishing a real MTM snapshot ~12% into the
    (tight, 0.2) margin_utilization_cap. TICK3..TICK6 are then offered as a
    4-candidate batch on day2: `size_selected()`'s margin-utilization-TARGET
    stage (not just a ceiling -- see allocation.py's Stage 2.5) now
    proactively scales all four DOWN from their own 30% Stage-1 cap to fit
    the ~8% of margin budget TICK1/TICK2 left remaining, rather than sizing
    them at 30% and letting the downstream admission gate reject the
    excess -- so `margin_rejected_count` is correctly 0 here (everything
    offered gets admitted, just at a smaller size), which is itself proof
    the target is load-bearing, not vacuously satisfied. Candidates use
    base_win_rate=0.99 (see `_candidate`'s docstring) so Kelly sizing
    saturates the leveraged 30% per-position cap cleanly, so any wave-2
    ticker sized below that must have been the margin target doing it, not
    Kelly alone.
    """
    day0 = datetime(2024, 1, 1)
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)

    wave1 = ["TICK1", "TICK2"]
    wave2 = ["TICK3", "TICK4", "TICK5", "TICK6"]
    all_tickers = wave1 + wave2

    dated_rows = _dated_rows([
        (day0, [_candidate(t, 100.0, 90.0, 140.0, base_win_rate=0.99) for t in wave1]),
        (day1, [_candidate(t, 100.0, 90.0, 140.0, base_win_rate=0.99) for t in wave2]),
        (day2, []),
    ])

    # Flat, no-SL/TP-touch bars for every ticker on every day it might be
    # queried (open positions + newly offered candidates), so nothing
    # closes early and every fill lands exactly at entry=100.
    bars_by_ticker = {
        t: {
            day1.date(): (100.0, 101.0, 99.0, 100.0, 1000.0),
            day2.date(): (100.0, 101.0, 99.0, 100.0, 1000.0),
        }
        for t in all_tickers
    }

    metrics, meta, client = _run_main(
        monkeypatch, tmp_path, dated_rows, bars_by_ticker,
        argv_extra=[
            "--capital", str(CAPITAL_2), "--top-n", "6",
            "--max-leverage", "2.0", "--margin-utilization", "0.2",
        ],
        account_name="exposure_cap_test",
    )
    try:
        margin_config = load_margin_config(MARGIN_CONFIG_PATH)
        classes = {t: classify_symbol(t, margin_config) for t in all_tickers}
        # Explicitly computed, not assumed: every synthetic ticker here
        # happens to fall into the same default bucket, but the min() is
        # taken generically over whatever classes actually appear.
        min_initial_margin_pct_fraction = min(c.initial_margin_pct for c in classes.values()) / 100.0
        assert min_initial_margin_pct_fraction == pytest.approx(0.20)

        rows = client._conn.execute(
            "SELECT date, equity, gross_notional FROM kairos_mtm_daily "
            "WHERE account_name = ? ORDER BY date",
            ("exposure_cap_test",),
        ).fetchall()
        assert rows, "expected at least one kairos_mtm_daily row"

        peak_date, peak_equity, peak_gross_notional = max(rows, key=lambda r: r[2])
        assert peak_gross_notional > 0.0

        # Exact ceiling if a day's margin usage sat AT margin_utilization_cap
        # (0.2) precisely: margin_used = notional * margin_pct, so
        # notional_at_cap = equity * cap / margin_pct. (The old formula wrote
        # this as `peak_equity * max_leverage=2.0 * 0.1 / margin_pct`, where
        # 2.0*0.1 was just an obscure way of spelling the same 0.2 cap.)
        # Stage 2.5 sizes each day off the PRIOR day's persisted equity
        # snapshot, not today's own -- so a small overshoot past this exact
        # ceiling is expected (today's own equity has already drifted from
        # yesterday's by entry costs before the notional is even placed), not
        # a bound violation. 10% tolerance still catches a real blowup (this
        # test caught a 245%-of-cap live bug from an equity-basis mismatch
        # before the admission_snapshot.equity fix -- see kairos_papertrade.py
        # around `alloc_config = AllocationConfig(...)`).
        notional_at_cap = peak_equity * 0.2 / min_initial_margin_pct_fraction
        bound = notional_at_cap * 1.10
        assert peak_gross_notional <= bound, (
            f"peak gross_notional={peak_gross_notional} on {peak_date} exceeds bound={bound} "
            f"(110% of exact at-cap notional {notional_at_cap} for "
            f"equity={peak_equity}, margin_utilization_cap=0.2, "
            f"min_initial_margin_pct={min_initial_margin_pct_fraction})"
        )

        # allocation.py's margin-utilization-TARGET stage (Stage 2.5) means
        # the cap is now load-bearing via proactive sizing, not downstream
        # rejection -- everything offered gets admitted (0 rejections), but
        # wave 2 gets scaled DOWN from its own Kelly-saturated 30% Stage-1
        # cap to fit the ~8% of margin budget wave 1 left remaining.
        assert meta["margin_rejected_count"] == 0
        assert meta["phantom_fill_rejected_count"] == 0

        # Positions still open at window end are removed (not force-closed --
        # see remove_all_open_positions), so the sizing itself is verified via
        # the persisted kairos_mtm_daily snapshots instead of live position
        # rows. Day 1 (wave 1 alone, nothing else using margin yet): each of
        # TICK1/TICK2 sizes at Kelly's own 30% Stage-1 cap, untouched by
        # Stage 2.5, so day 1's initial_margin_used is ~12% of equity (2 *
        # 30% * 20% margin_pct = 12% before entry costs shave equity down
        # slightly; 0.121071... confirmed empirically, pinned exactly like
        # this file's other hand-derived values).
        by_date = {r[0]: r for r in client._conn.execute(
            "SELECT date, equity, gross_notional, initial_margin_used, margin_utilization "
            "FROM kairos_mtm_daily WHERE account_name = ? ORDER BY date",
            ("exposure_cap_test",),
        ).fetchall()}
        day1_row = by_date["2024-01-02"]
        assert day1_row[4] == pytest.approx(0.1210714826211976, rel=1e-9)  # margin_utilization

        # Day 2 (wave 1 + wave 2 together): Stage 2.5 targets the day's
        # REMAINING headroom (0.2 - 0.121071... ~= 0.079) for wave 2, sized
        # in PERCENTAGE-OF-EQUITY terms against day 1's *persisted*
        # admission_snapshot.equity (not phantom's raw account.cash -- using
        # raw cash here was a real bug, fixed in kairos_papertrade.py: the
        # two can diverge by more than the whole starting capital once
        # shorts/financing are involved, which silently blew Stage 2.5's
        # target up to 245% of cap in a live run). Day 2's own
        # kairos_mtm_daily margin_utilization is still computed against a
        # slightly different, TODAY's-own equity figure (corrected_cash +
        # unrealized_pnl, drifted from day 1's persisted snapshot by entry
        # costs/financing accrual even with flat, non-moving bars) -- so the
        # ACHIEVED utilization (0.20282..., pinned exactly, confirmed
        # empirically) lands close to, and here just barely over, the 20%
        # target rather than bit-for-bit on it. That small residual gap is
        # inherent to the system using more than one cash/equity tracker
        # (see docs/papertrade_loss_analysis.md and the 10%-tolerance bound
        # check above), not a sizing bug -- what matters here is that it's
        # dramatically closer to the 20% target than the pre-Stage-2.5
        # behavior would have left it (~0.121, since Kelly alone never asked
        # for more and nothing used to scale allocations up to fill the rest
        # of the budget).
        day2_row = by_date["2024-01-03"]
        assert day2_row[4] == pytest.approx(0.20282169644247902, rel=1e-9)  # margin_utilization
        assert day2_row[4] > day1_row[4]  # meaningfully closer to the 20% target than day 1 was
    finally:
        client._conn.close()


# =============================================================================
# Test 3: BUG-01 -- same-day fill+close round trip
# =============================================================================

CAPITAL_3 = 10000.0


def test_same_day_round_trip_reflected_in_corrected_cash(monkeypatch, tmp_path):
    """BUG-01 regression: a position whose fill-day bar's High already
    crosses `target` on THAT SAME bar (a stop/target hit on the day it
    fills -- routine for tight stops on volatile daily-bar assets) must
    still have its P&L applied to `corrected_cash`/`kairos_mtm_daily`.

    Before the fix, the day-loop's fill/close diffing only compared
    `current_open` (queried AFTER runner.backtest() returns, by which point
    this position is already `status='closed'`) against `known_open_ids`
    (which never saw it, since it didn't exist before this same iteration)
    -- so it was invisible to BOTH loops, and `corrected_cash`/
    `kairos_mtm_daily` stayed flat at capital despite a real closed,
    profitable trade. This is the exact confirmed-live scenario from
    docs/tickets/BUG-01-same-day-fill-close-blind-spot.md.

    Numbers reuse test_leverage_off_matches_pinned_baseline's TICKA
    derivation verbatim (same entry/stop/target/qty/exit price) -- only the
    SL/TP touch is moved onto the fill bar itself instead of the following
    day: corrected_realized_pnl = 594.70.
    """
    day0 = datetime(2024, 1, 1)
    day1 = datetime(2024, 1, 2)

    dated_rows = _dated_rows([
        (day0, [_candidate("TICKA", 100.0, 90.0, 140.0)]),
        (day1, []),
    ])

    bars_by_ticker = {
        "TICKA": {
            # Open=100 fills the entry; High=145 >= target(140) triggers a
            # take-profit close on THIS SAME bar/day -- the ticket's exact
            # suggested repro shape.
            day1.date(): (100.0, 145.0, 95.0, 130.0, 1000.0),
        },
    }

    metrics, meta, client = _run_main(
        monkeypatch, tmp_path, dated_rows, bars_by_ticker,
        argv_extra=[
            "--capital", str(CAPITAL_3), "--top-n", "3",
            "--max-leverage", "1.0", "--margin-utilization", "0.8",
        ],
        account_name="same_day_round_trip",
    )
    try:
        closed = client.positions.list(account_name="same_day_round_trip", status="closed")
        assert len(closed) == 1
        pos = closed[0]
        assert pos.close_reason == "tp"
        assert pos.entry_datetime.date() == pos.exit_datetime.date() == day1.date()

        expected_profit = 594.70
        assert metrics["total_profit_eur"] == pytest.approx(expected_profit, rel=1e-9)
        assert metrics["num_trades"] == 1

        rows = client._conn.execute(
            "SELECT date, cash, equity, gross_notional FROM kairos_mtm_daily "
            "WHERE account_name = ? ORDER BY date",
            ("same_day_round_trip",),
        ).fetchall()
        assert rows, "expected at least one kairos_mtm_daily row"

        # THE bug: pre-fix, every row stays flat at capital (the round trip
        # is invisible to corrected_cash) -- these are the assertions that
        # fail on unfixed code.
        final_cash, final_equity = rows[-1][1], rows[-1][2]
        assert final_cash != pytest.approx(CAPITAL_3)
        assert final_equity == pytest.approx(CAPITAL_3 + expected_profit, rel=1e-9)

        # Ticket DoD #3: kairos_mtm_daily's final equity must reconcile with
        # the closed-trade equity curve -- same convergence invariant as
        # test_kairos_papertrade_mtm_repro.py::
        # test_final_mtm_equity_equals_final_closed_trade_equity.
        assert metrics["mtm_total_return_pct"] == pytest.approx(metrics["pct_profit"], rel=1e-6)
    finally:
        client._conn.close()


# =============================================================================
# Test 4: BUG-01 -- same-day round trip mixed with an ordinary multi-day hold
# =============================================================================

CAPITAL_4 = 10000.0


def test_same_day_round_trip_mixed_with_multiday_position(monkeypatch, tmp_path):
    """Regression companion to the pure same-day-round-trip test above:
    proves the BUG-01 fix's new "closed position never seen open" loop
    does not double-count or otherwise disturb the EXISTING (already-
    working) multi-day fill/close path when both kinds of trades appear in
    the same run.

    TICKA: same-day round trip on day1 (fills + take-profits on the same
    bar, exactly as in test_same_day_round_trip_reflected_in_corrected_cash).
    TICKB: ordinary multi-day hold -- fills day1, held flat through day2
    (which forces the day-loop's diffing block to run AGAIN with TICKA
    already closed -- the exact condition that would double-count TICKA
    under a naive "status=closed, not in known_open_ids" check with no date
    filter, since `positions.list(status="closed")` returns every closed
    position ever, not just the current day's), stop-losses on day3.

    Both positions individually reuse test_leverage_off_matches_pinned_
    baseline's hand-derived P&L (TICKA's TP number now realized same-day
    instead of next-day; TICKB's SL number unchanged), so the combined
    total is the same pinned constant that test asserts
    (EXPECTED_METRICS_LEVERAGE_OFF), even though the trade timing differs.

    Note on `kairos_mtm_daily` cash/equity below: both tickers are plain
    (non-"-USD") symbols, so they classify as `equity_cfd` in
    config/margin_ibkr.yaml (initial_margin_pct=20, NOT spot -- true even
    at --max-leverage 1.0, see CLAUDE.md's "Configurable signal selection"
    / margin gotchas) and accrue overnight financing on TICKB while it's
    held. That financing is debited from `corrected_cash` only (a
    Kairos-side-only accrual; phantom's raw `account.cash` never sees it,
    same as a real CFD financing charge), so it does NOT feed into
    `total_profit_eur`/`pct_profit` (computed purely from closed positions'
    `realized_pnl`) -- it's an intentional, pre-existing, orthogonal source
    of cash/equity divergence, not a BUG-01 symptom. The cash/equity values
    pinned below are hand-verified to include it (see inline derivation).
    """
    day0 = datetime(2024, 1, 1)
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)
    day3 = datetime(2024, 1, 4)

    dated_rows = _dated_rows([
        (day0, [_candidate("TICKA", 100.0, 90.0, 140.0), _candidate("TICKB", 50.0, 45.0, 65.0)]),
        (day1, []),
        (day2, []),
        (day3, []),
    ])

    bars_by_ticker = {
        "TICKA": {
            # Same-day round trip: fills at Open=100, High=145 >= target(140)
            # triggers TP on this same bar.
            day1.date(): (100.0, 145.0, 95.0, 130.0, 1000.0),
        },
        "TICKB": {
            day1.date(): (50.0, 50.5, 49.5, 50.2, 1000.0),  # fills at Open=50, no touch
            day2.date(): (50.0, 50.5, 49.5, 50.2, 1000.0),  # still open, no touch -- forces the
                                                              # day-loop to run its diffing block
                                                              # again with TICKA already closed
            day3.date(): (48.0, 49.0, 44.0, 44.5, 1000.0),  # Low<=45 -> sl @ 45 exactly
        },
    }

    metrics, meta, client = _run_main(
        monkeypatch, tmp_path, dated_rows, bars_by_ticker,
        argv_extra=[
            "--capital", str(CAPITAL_4), "--top-n", "3",
            "--max-leverage", "1.0", "--margin-utilization", "0.8",
        ],
        account_name="same_day_plus_multiday",
    )
    try:
        closed = {
            p.ticker: p for p in client.positions.list(
                account_name="same_day_plus_multiday", status="closed",
            )
        }
        assert set(closed) == {"TICKA", "TICKB"}
        assert closed["TICKA"].entry_datetime.date() == closed["TICKA"].exit_datetime.date()
        assert closed["TICKB"].entry_datetime.date() != closed["TICKB"].exit_datetime.date()

        for key, expected in EXPECTED_METRICS_LEVERAGE_OFF.items():
            assert metrics[key] == pytest.approx(expected, rel=1e-9), key

        rows = {
            r[0]: r for r in client._conn.execute(
                "SELECT date, cash, equity, financing_accrued_day FROM kairos_mtm_daily "
                "WHERE account_name = ? ORDER BY date",
                ("same_day_plus_multiday",),
            ).fetchall()
        }
        assert set(rows) == {"2024-01-02", "2024-01-03", "2024-01-04"}

        # day1 (TICKA's round-trip day): corrected_cash = capital + TICKA's
        # fill+close net (594.70, the identity fill_delta+close_delta ==
        # corrected_realized_pnl) - TICKB's fill debit (1500 notional + 3.25
        # entry costs) - TICKB's day1 overnight financing (1506 notional_close
        # * (3.15+1.5)/360 == 19.4525). This is the assertion that fails
        # pre-fix: without the fix, TICKA's round trip contributes nothing,
        # leaving cash at capital - 1503.25 - 19.4525 == 8477.2975 instead.
        assert rows["2024-01-02"][1] == pytest.approx(9071.9975, rel=1e-9)
        assert rows["2024-01-02"][3] == pytest.approx(19.4525, rel=1e-9)

        # day2: TICKB held flat (no fill/close event) -- exercises the fix's
        # date filter, since TICKA is STILL in positions.list(status="closed")
        # here but must NOT be reprocessed. One more night of TICKB financing.
        assert rows["2024-01-03"][1] == pytest.approx(9052.545, rel=1e-9)
        assert rows["2024-01-03"][3] == pytest.approx(19.4525, rel=1e-9)

        # day3: TICKB closes (sl). Final cash must equal capital + both
        # trades' combined corrected P&L (439.775) minus the two nights of
        # TICKB financing (38.905) -- i.e. exactly phantom's raw closed-trade
        # equity (10439.775) minus financing, not flat and not double-counted.
        assert rows["2024-01-04"][1] == pytest.approx(10400.87, rel=1e-9)
        assert rows["2024-01-04"][2] == pytest.approx(10400.87, rel=1e-9)
    finally:
        client._conn.close()


# =============================================================================
# Test 4b: BUG-11 -- same-day round trip must not be re-counted across the
# multiple hourly day-loop iterations --interval 1h shares per calendar day
# =============================================================================

CAPITAL_6 = 10000.0


def test_same_day_round_trip_not_recounted_across_hourly_iterations(monkeypatch, tmp_path):
    """BUG-11 regression: the same-day-round-trip scan (see
    `_fill_cash_delta`/`_close_cash_delta` callers in kairos_papertrade.py's
    main() day loop) re-lists EVERY closed position each iteration and keeps
    any that closed "today" -- with `--interval 1d` there is only ever one
    iteration per calendar day, so this is harmless. With `--interval 1h`
    (or, as reproduced here, several dated_rows sharing one calendar date),
    the SAME round-trip position matches this "closed today" filter again on
    every later iteration of that same day, re-applying its fill+close cash
    delta each time -- silently multiplying its P&L's contribution to
    `corrected_cash` by however many same-day iterations follow.

    Identical scenario to test_same_day_round_trip_mixed_with_multiday_position
    (TICKA same-day round trip + TICKB ordinary multi-day hold, same bars,
    same capital) EXCEPT day1 is split into four hourly iterations
    (day1_h0..h3) instead of one daily iteration. TICKB stays open through
    all four, so `tickers` is non-empty on h1/h2/h3 too (see kairos_papertrade
    .py's `if tickers:` gate) -- meaning the buggy scan actually runs on
    those extra iterations, unlike a naive same-ticker-only reproduction
    where an empty `tickers` set would skip the block entirely and hide the
    bug. Pre-fix, h1/h2/h3 each re-add TICKA's already-counted 594.70 fill+
    close delta to `corrected_cash`; post-fix (and in the sibling daily-
    interval test), every downstream number must be IDENTICAL to that
    sibling test's pinned values, since de-duping same-day round trips must
    not depend on how many hourly iterations happen to share a calendar day.
    """
    day0 = datetime(2024, 1, 1)
    day1_h0 = datetime(2024, 1, 2, 0)
    day1_h1 = datetime(2024, 1, 2, 1)
    day1_h2 = datetime(2024, 1, 2, 2)
    day1_h3 = datetime(2024, 1, 2, 3)
    day2 = datetime(2024, 1, 3)
    day3 = datetime(2024, 1, 4)

    dated_rows = _dated_rows([
        (day0, [_candidate("TICKA", 100.0, 90.0, 140.0), _candidate("TICKB", 50.0, 45.0, 65.0)]),
        (day1_h0, []),
        (day1_h1, []),
        (day1_h2, []),
        (day1_h3, []),
        (day2, []),
        (day3, []),
    ])

    bars_by_ticker = {
        "TICKA": {
            # Open=100 fills the entry; High=145 >= target(140) triggers a
            # take-profit close on THIS SAME bar/day. day1_h0..h3 all query
            # this same day1 bar (day_start/day_end are derived from the
            # calendar date alone), but TICKA is already closed after h0, so
            # h1/h2/h3 are no-op fills/closes for phantom itself -- the bug
            # is entirely in Kairos-side corrected_cash bookkeeping
            # re-scanning an already-processed closed position, not in
            # phantom's own state.
            day1_h0.date(): (100.0, 145.0, 95.0, 130.0, 1000.0),
        },
        "TICKB": {
            # Fills day1 (Open=50, no touch) and stays open through h0-h3 --
            # this is what keeps `tickers` non-empty on h1/h2/h3 so the
            # buggy scan actually executes on them (see docstring).
            day1_h0.date(): (50.0, 50.5, 49.5, 50.2, 1000.0),
            day2.date(): (50.0, 50.5, 49.5, 50.2, 1000.0),
            day3.date(): (48.0, 49.0, 44.0, 44.5, 1000.0),  # Low<=45 -> sl @ 45 exactly
        },
    }

    metrics, meta, client = _run_main(
        monkeypatch, tmp_path, dated_rows, bars_by_ticker,
        argv_extra=[
            "--capital", str(CAPITAL_6), "--top-n", "3",
            "--max-leverage", "1.0", "--margin-utilization", "0.8",
        ],
        account_name="same_day_round_trip_hourly",
    )
    try:
        closed = {
            p.ticker: p for p in client.positions.list(
                account_name="same_day_round_trip_hourly", status="closed",
            )
        }
        assert set(closed) == {"TICKA", "TICKB"}

        # Identical to EXPECTED_METRICS_LEVERAGE_OFF / test 4's totals --
        # splitting day1 into four hourly iterations must not change the
        # real, phantom-tracked outcome at all.
        for key, expected in EXPECTED_METRICS_LEVERAGE_OFF.items():
            assert metrics[key] == pytest.approx(expected, rel=1e-9), key

        rows = {
            r[0]: r for r in client._conn.execute(
                "SELECT date, cash, equity, financing_accrued_day FROM kairos_mtm_daily "
                "WHERE account_name = ? ORDER BY date",
                ("same_day_round_trip_hourly",),
            ).fetchall()
        }
        assert set(rows) == {"2024-01-02", "2024-01-03", "2024-01-04"}

        # THE bug: pre-fix, h1/h2/h3 each re-add TICKA's 594.70 round-trip
        # delta to corrected_cash on top of day1's own (already-persisted,
        # correct) row -- so day2's row, persisted from the carried-forward
        # corrected_cash, would read ~9052.545 + 3*594.70 instead of exactly
        # matching test 4's pinned day2/day3 values below.
        assert rows["2024-01-02"][1] == pytest.approx(9071.9975, rel=1e-9)
        assert rows["2024-01-03"][1] == pytest.approx(9052.545, rel=1e-9)
        assert rows["2024-01-04"][1] == pytest.approx(10400.87, rel=1e-9)
        assert rows["2024-01-04"][2] == pytest.approx(10400.87, rel=1e-9)
        # NOTE: unlike the pure-round-trip test above, mtm_total_return_pct is
        # NOT expected to equal pct_profit here -- TICKB's overnight financing
        # is an intentional, orthogonal cash/equity divergence (see test
        # test_same_day_round_trip_mixed_with_multiday_position's docstring),
        # not something this test is checking. The three row assertions above,
        # matching that sibling daily-interval test's pinned values exactly,
        # are what prove hourly splitting doesn't change the outcome.
    finally:
        client._conn.close()


# =============================================================================
# Test 5: RESEARCH-01 -- stale stop/target bracket rejected before order placement
# =============================================================================

CAPITAL_5 = 10000.0


def test_stale_bracket_order_is_skipped_not_placed(monkeypatch, tmp_path):
    """RESEARCH-01 regression: a candidate whose `stop`/`target` were computed
    correctly relative to a NOW-STALE reference price must never reach
    phantom as an `Order(...)` once the fill day's real price has moved past
    those levels.

    Root cause (docs/tickets/RESEARCH-01-sl-close-reason-positive-pnl.md): a
    live run showed 3 WLD-USD long positions closed `close_reason='sl'` with
    POSITIVE `realized_pnl` and an `exit_price` above `entry_price` -- the
    exact opposite of what a stop-loss should do. Root-caused via the run's
    own `data/pipeline_results.db` (`signals_cache` table) and two SEPARATE
    local `price_cache` SQLite mirrors: `data/yfd_prices.db` (used by
    `kairos_signals`/`fetch_data_raw` for SIGNAL generation) had silently
    stopped advancing for WLD-USD after 2026-07-25 (a stale `no_data_tickers`
    marker), so `HighLowStrategy.generate_signal` kept computing
    stop=l*0.99/target=h from the SAME frozen current_price for FOUR straight
    as_of dates (confirmed via `signals_cache.stats_json`, byte-identical
    across 07-25..07-28). Meanwhile `_IntradayFallbackProvider`'s OWN,
    separate mirror (`data/phantom_ledger/yfd_prices.db`) kept getting fresh
    bars, so the real fill price kept dropping while the order's fixed
    absolute `stop_loss` stayed pinned above it. phantom's
    `PositionManager.determine_close` (see `.venv/.../phantom/engine/
    position_manager.py`) triggers purely off `bar.Low <= stop_loss` and
    reports `exit_price = stop_loss` verbatim -- so the long closed "sl" on
    the very same day (also BUG-01's same-day pattern) with exit > entry.

    Fix: `kairos_papertrade.py`'s day loop now fetches each candidate
    ticker's OWN fill-day Open via `intraday_provider.get_bars()` (the exact
    same call phantom's `runner.backtest()` is about to make) BEFORE
    constructing the `Order`, and skips (does not place) any candidate whose
    stop/target no longer bracket that fresh price on the correct side (see
    `_bracket_is_stale`). This scenario reproduces the mechanism directly:
    TICKA's candidate (entry=100, stop=90, target=140) was sane relative to
    its OWN signal-time reference (100), but the fill day's real Open has
    already crashed to 70 -- below stop=90 -- exactly like WLD-USD's real
    price outrunning its stale stop. Before the fix this would fill at 70
    and instantly "stop out" at exit_price=90 for a nonsensical +20 gain
    tagged 'sl'; after the fix, no order/position for TICKA should exist at
    all. TICKB is an ordinary, non-stale candidate in the SAME run, proving
    the guard doesn't reject healthy signals too -- it reuses
    test_leverage_off_matches_pinned_baseline's exact TICKB path (fills
    day1, stop-losses day2) so it actually CLOSES before window end, rather
    than being silently refunded/dropped by `remove_all_open_positions`
    (main()'s normal end-of-window handling for any still-OPEN position,
    which would otherwise make an "open forever" TICKB indistinguishable
    from a wrongly-skipped one in this test).
    """
    day0 = datetime(2024, 1, 1)
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)

    dated_rows = _dated_rows([
        (day0, [
            _candidate("TICKA", 100.0, 90.0, 140.0),  # sane vs its OWN signal-time reference (100)
            _candidate("TICKB", 50.0, 45.0, 65.0),    # ordinary, stays sane at fill time too
        ]),
        (day1, []),
        (day2, []),
    ])

    bars_by_ticker = {
        "TICKA": {
            # Real price has crashed to 70 by fill day -- stop=90 sits ABOVE
            # this Open, so the bracket is stale (would trigger an instant,
            # mislabeled "sl" win if placed). No entry for day2: TICKA must
            # never fill, so no later bar is needed.
            day1.date(): (70.0, 72.0, 68.0, 69.0, 1000.0),
        },
        "TICKB": {
            day1.date(): (50.0, 50.5, 49.5, 50.2, 1000.0),   # fills at Open=50, no touch
            day2.date(): (48.0, 49.0, 44.0, 44.5, 1000.0),   # Low<=45 -> sl @ 45 exactly
        },
    }

    metrics, meta, client = _run_main(
        monkeypatch, tmp_path, dated_rows, bars_by_ticker,
        argv_extra=[
            "--capital", str(CAPITAL_5), "--top-n", "3",
            "--max-leverage", "1.0", "--margin-utilization", "0.8",
        ],
        account_name="stale_bracket_test",
    )
    try:
        # TICKA: the stale-bracket candidate must never become an order or a
        # position -- this is the actual fix under test.
        ticka_orders = [
            o for o in client.orders.list(account_name="stale_bracket_test", status=None)
            if o.ticker == "TICKA"
        ]
        ticka_positions = [
            p for p in client.positions.list(account_name="stale_bracket_test", status=None)
            if p.ticker == "TICKA"
        ]
        assert ticka_orders == []
        assert ticka_positions == []

        # TICKB: an ordinary, non-stale candidate in the same batch must
        # still fill and close normally -- proves the guard is targeted,
        # not a blanket order-placement regression.
        tickb_positions = [
            p for p in client.positions.list(account_name="stale_bracket_test", status=None)
            if p.ticker == "TICKB"
        ]
        assert len(tickb_positions) == 1
        assert tickb_positions[0].entry_price == pytest.approx(50.0)
        assert tickb_positions[0].close_reason == "sl"

        assert meta["margin_rejected_count"] == 0
        assert metrics["num_trades"] == 1
    finally:
        client._conn.close()


# =============================================================================
# Test 6: BUG-02 -- admission check must see same-day round trips' margin usage
# =============================================================================

CAPITAL_6 = 10000.0


def test_admission_check_counts_same_day_round_trip_margin(monkeypatch, tmp_path):
    """BUG-02 regression: `admission_check`'s gate for a NEW batch of orders
    must be based on margin usage that reflects prior same-day round-trip
    trades, not just positions still `status='open'` when a `DailySnapshot`
    is taken.

    Root cause: `last_snapshot` (what `admission_check` sees) is built from
    `mtm_positions`, which is sourced from `current_open` positions only
    (kairos_papertrade.py's day loop, ~line 2203). BUG-01's fix applies
    same-day round trips' P&L to `corrected_cash` but never added them to
    `mtm_positions` -- confirmed empirically (see this ticket) by probing
    `kairos_mtm_daily.initial_margin_used`/`gross_notional` after BUG-01's
    fix alone: they stayed at 0.0 for a day containing only same-day round
    trips. So `admission_check`'s `new_initial_margin_used` was computed
    against a permanently-zero baseline for accounts dominated by such
    trades, defeating `margin_utilization_cap` -- exactly the scenario in
    BUG-01's confirmed live repro (3/3 real trades were same-day round
    trips).

    Scenario (ticket's suggested structure): wave 1 (TICK1, TICK2) is
    offered on day0 -- admitted unchecked (first-ever batch, `last_snapshot`
    is None) -- and same-day round-trips (fills + take-profits) on day1,
    each consuming ~600 EUR of initial margin (30%-of-cash leveraged
    Kelly-capped alloc -- max_leverage=2.0 doubles the legacy 15% cap to
    30% for equity_cfd's 20% margin class, see `_candidate`'s docstring --
    * 20% equity_cfd initial_margin_pct). Wave 2 (TICK3..TICK6) is then
    offered on day1 too, so it is admission-checked against `last_snapshot`
    from day1 -- the same snapshot day1's round trips should have fed into.
    `margin_utilization` is set tight enough (0.26) that wave 2's aggregate
    notional fits entirely if day1's (already-closed) usage is ignored (0
    rejected -- the pre-fix/BUG-02 symptom, confirmed empirically) but must
    partially breach the cap once day1's real usage is correctly counted
    (some but not all of wave 2 rejected, confirmed empirically on the
    fixed code). Candidates use base_win_rate=0.99 (see `_candidate`'s
    docstring) so Kelly sizing saturates the leveraged 30% cap cleanly.
    """
    day0 = datetime(2024, 1, 1)
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)

    wave1 = ["TICK1", "TICK2"]
    wave2 = ["TICK3", "TICK4", "TICK5", "TICK6"]
    all_tickers = wave1 + wave2

    dated_rows = _dated_rows([
        (day0, [_candidate(t, 100.0, 90.0, 140.0, base_win_rate=0.99) for t in wave1]),
        (day1, [_candidate(t, 100.0, 90.0, 140.0, base_win_rate=0.99) for t in wave2]),
        (day2, []),
    ])

    bars_by_ticker = {
        t: {
            # wave1 tickers same-day round trip on day1: fill at Open=100,
            # High=145 crosses target=140 on the SAME bar (BUG-01's exact
            # repro shape). wave2 tickers get a flat, no-touch bar on day1
            # (their fill day) so they stay open into day2.
            day1.date(): (
                (100.0, 145.0, 95.0, 130.0, 1000.0) if t in wave1
                else (100.0, 101.0, 99.0, 100.0, 1000.0)
            ),
            day2.date(): (100.0, 101.0, 99.0, 100.0, 1000.0),
        }
        for t in all_tickers
    }

    metrics, meta, client = _run_main(
        monkeypatch, tmp_path, dated_rows, bars_by_ticker,
        argv_extra=[
            "--capital", str(CAPITAL_6), "--top-n", "6",
            "--max-leverage", "2.0", "--margin-utilization", "0.26",
        ],
        account_name="bug02_admission_test",
    )
    try:
        closed = client.positions.list(account_name="bug02_admission_test", status="closed")
        wave1_closed = [p for p in closed if p.ticker in wave1]
        assert len(wave1_closed) == 2
        for p in wave1_closed:
            assert p.entry_datetime.date() == p.exit_datetime.date() == day1.date()

        rows = {
            r[0]: r for r in client._conn.execute(
                "SELECT date, gross_notional, initial_margin_used FROM kairos_mtm_daily "
                "WHERE account_name = ? ORDER BY date",
                ("bug02_admission_test",),
            ).fetchall()
        }
        assert set(rows) >= {"2024-01-02", "2024-01-03"}

        # THE fix: day1's snapshot (== tomorrow's `last_snapshot` for
        # admission_check) must reflect wave1's round-trip margin usage, not
        # stay at 0.0 -- this is the exact BUG-02 symptom confirmed
        # empirically against BUG-01-fixed-but-BUG-02-unfixed code.
        day1_gross_notional, day1_initial_margin = rows["2024-01-02"][1], rows["2024-01-02"][2]
        assert day1_gross_notional == pytest.approx(6000.0, rel=1e-9)
        assert day1_initial_margin == pytest.approx(1200.0, rel=1e-9)

        # The cap must actually have been load-bearing for wave 2 (not
        # vacuously satisfied): at least one, but not all, of wave 2's 4
        # candidates rejected. Pre-fix this is 0 (all 4 admitted) because
        # day1's round-trip usage was invisible to the gate.
        assert meta["margin_rejected_count"] > 0
        assert meta["margin_rejected_count"] < len(wave2)
    finally:
        client._conn.close()


# =============================================================================
# Test 7: _sync_margin_classes harmonizes phantom's broker profile
# =============================================================================

def test_sync_margin_classes_matches_config(monkeypatch, tmp_path):
    """phantom_ledger E17-S01 added per-instrument-class CFD margin rates, but
    phantom's bundled ibkr.json profile uses IBKR-native symbol spellings
    ("EURUSD", "XAUUSD", "BTC") that can never match Kairos's yfinance-style
    tickers ("EURUSD=X", "GC=F", "BTC-USD") under `MarginModel.margin_pct_for()`
    (re.fullmatch). `_sync_margin_classes` patches the loaded broker profile
    with Kairos's own `config/margin_ibkr.yaml` classes so the per-class rates
    actually apply. Drives a real (minimal) `main()` invocation -- which
    already calls `_sync_margin_classes` as part of its setup -- rather than
    calling it directly, so this also proves it's correctly wired in.
    """
    day0 = datetime(2024, 1, 1)
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)

    dated_rows = _dated_rows([
        (day0, [_candidate("TICKA", 100.0, 90.0, 140.0)]),
        (day1, []),
        (day2, []),
    ])
    bars_by_ticker = {
        "TICKA": {
            day1.date(): (100.0, 101.0, 99.0, 100.5, 1000.0),
            day2.date(): (105.0, 141.0, 104.0, 138.0, 1000.0),
        },
    }

    _metrics, _meta, client = _run_main(
        monkeypatch, tmp_path, dated_rows, bars_by_ticker,
        argv_extra=["--capital", "10000", "--top-n", "3", "--max-leverage", "1.0"],
        account_name="sync_margin_classes_test",
    )
    try:
        margin_config = load_margin_config(MARGIN_CONFIG_PATH)
        profile = client.brokers.get("IBKR")

        # equity_cfd (Kairos's catch-all, match=None/symbols=None) becomes
        # phantom's default_margin_pct, not a MarginClassRule -- no longer
        # the bundled profile's original 0.25.
        assert profile.margin.default_margin_pct == pytest.approx(0.20)

        by_label = {rule.label: rule for rule in profile.margin.classes}
        assert set(by_label) == {
            "fx_major", "fx_minor", "index_gold_major", "commodity_other", "crypto_spot",
        }
        # crypto_cfd (enabled: false in the YAML) must be skipped entirely.
        assert "crypto_cfd" not in by_label

        # Percentage-points -> fraction conversion.
        assert by_label["fx_major"].margin_pct == pytest.approx(0.0333, rel=1e-6)
        assert by_label["commodity_other"].margin_pct == pytest.approx(0.10)
        assert by_label["crypto_spot"].margin_pct == pytest.approx(1.0)

        # Explicit-symbols class carried through unchanged (as a sorted list).
        assert by_label["fx_major"].symbols == sorted(margin_config.classes["fx_major"].symbols)
        assert by_label["fx_major"].match is None

        # Regex classes: Kairos's re.search-style, front-unanchored pattern
        # must be translated for phantom's re.fullmatch -- the untranslated
        # original pattern must NOT fullmatch a real ticker (proving the `.*`
        # prefix is load-bearing, not cosmetic), while the translated one must.
        original_pattern = margin_config.classes["fx_minor"].match
        assert original_pattern == "=X$"
        assert re.fullmatch(original_pattern, "EURJPY=X") is None
        assert by_label["fx_minor"].match == ".*=X$"
        assert re.fullmatch(by_label["fx_minor"].match, "EURJPY=X") is not None
        assert by_label["commodity_other"].match == ".*=F$"
        assert re.fullmatch(by_label["commodity_other"].match, "GC=F") is not None
        assert by_label["crypto_spot"].match == ".*-USD$"
        assert re.fullmatch(by_label["crypto_spot"].match, "BTC-USD") is not None

        # margin_pct_for() actually resolves a Kairos ticker now (pre-fix this
        # always fell through to default_margin_pct=0.25 regardless of class).
        assert profile.margin.margin_pct_for("BTC-USD") == pytest.approx(1.0)
        # GC=F is an explicit symbol in index_gold_major (5%), not
        # commodity_other's =F$ regex fallback (10%) -- explicit symbols win.
        assert profile.margin.margin_pct_for("GC=F") == pytest.approx(0.05)
        assert profile.margin.margin_pct_for("CL=F") == pytest.approx(0.10)
        assert profile.margin.margin_pct_for("EURUSD=X") == pytest.approx(0.0333, rel=1e-6)

        # Idempotency: a second call with the same config must not touch the
        # DB row (same broker profile `id`, byte-identical config_json).
        row_before = client._conn.execute(
            "SELECT id, config_json FROM broker_profiles WHERE name = ?", ("IBKR",)
        ).fetchone()
        kairos_papertrade._sync_margin_classes(client, "IBKR", margin_config)
        row_after = client._conn.execute(
            "SELECT id, config_json FROM broker_profiles WHERE name = ?", ("IBKR",)
        ).fetchone()
        assert row_after["id"] == row_before["id"]
        assert row_after["config_json"] == row_before["config_json"]
    finally:
        client._conn.close()


# =============================================================================
# Test 8: BUG-03 -- the first-ever admission-gated batch must actually be gated
# =============================================================================

def test_first_batch_is_admission_gated_not_skipped():
    """`main()`'s day loop used to pass `snapshot=None` into
    `_place_batch_orders` on the very first iteration that places orders
    (`last_snapshot` starts as `None`), which `_place_order_if_admitted`
    treats as "skip the check entirely". Fine for a single order, but a
    multi-position selection rule (e.g. TOP 8) can blow past
    `margin_utilization_cap` in that one ungated batch alone -- confirmed on
    a real leveraged run where the first batch alone reached 174.6% margin
    utilization against an 80% cap. Fixed by building a zero-margin-used
    bootstrap `DailySnapshot` instead of passing `None` on that first
    iteration (see BUG-03 at the call site in `main()`).

    This test exercises `_place_batch_orders` directly (the function itself
    is unchanged -- only main()'s call site is), proving both halves: `None`
    still means "skip" (kept for other callers), and a bootstrap snapshot
    with `initial_margin_used=0.0` gates the batch exactly like every
    subsequent day would.
    """
    from unittest.mock import MagicMock
    from allocation import AllocationConfig
    from kairos_mtm import DailySnapshot

    margin_config = load_margin_config(MARGIN_CONFIG_PATH)
    alloc_config = AllocationConfig(max_leverage=5.0, margin_utilization_cap=0.8)

    # 3 plain-ticker (equity_cfd class, 20% margin) orders of notional=300
    # each: margin_needed = 300*0.20 = 60 per order. Against equity=200 and
    # an 80% cap (max_margin=160), orders 1+2 fit (60+60=120<=160) but order
    # 3 doesn't (120+60=180>160) -- IF the batch is actually gated.
    order_requests = [(f"order-{i}", f"TICK{i}", 300.0) for i in range(3)]

    # Pre-fix behavior: snapshot=None skips the check entirely -- all 3
    # admitted regardless of how far over the cap they'd push margin usage.
    client_skip = MagicMock()
    rejected_skip = kairos_papertrade._place_batch_orders(
        client_skip, "acct-1", order_requests, datetime(2024, 1, 1),
        None, margin_config, alloc_config,
    )
    assert rejected_skip == 0
    assert client_skip.orders.place.call_count == 3

    # Fixed behavior: a bootstrap DailySnapshot (equity=200, margin_used=0)
    # gates the SAME batch -- order 3 must be rejected.
    bootstrap_snapshot = DailySnapshot(
        date=datetime(2024, 1, 1).date(), cash=200.0, unrealized_pnl=0.0,
        equity=200.0, gross_notional=0.0, initial_margin_used=0.0,
        maintenance_margin_used=0.0, free_margin=200.0,
        margin_utilization=0.0, financing_accrued_day=0.0, liquidations=0,
    )
    client_gated = MagicMock()
    rejected_gated = kairos_papertrade._place_batch_orders(
        client_gated, "acct-1", order_requests, datetime(2024, 1, 1),
        bootstrap_snapshot, margin_config, alloc_config,
    )
    assert rejected_gated == 1
    assert client_gated.orders.place.call_count == 2


# =============================================================================
# Test 9: BUG-04 -- long stock/ETF positions get margined in phantom too,
# once leveraged
# =============================================================================

def test_map_instrument_type_routes_leveraged_stocks_to_cfd():
    """A long plain-equity/ETF ticker (no futures/forex/crypto suffix) used
    to always map to phantom's "stock" instrument type, regardless of
    --max-leverage -- phantom's handle_fill() only applies margin/leverage
    accounting for instrument_type=="cfd", so it always charged FULL
    notional cash for a "stock" fill even though ticker_max_leverage
    (kairos_margin.classify_symbol's equity_cfd class, 20% margin/5x) told
    allocation.py's sizing this ticker could be leveraged. Confirmed on a
    real leveraged run: every long stock/ETF position across all 14
    selection rules showed leverage=1.0/margin_required=0.0 in phantom
    regardless of --max-leverage, silently defeating leverage for any rule
    with meaningful long-equity exposure.

    map_instrument_type() now reuses _use_full_notional()'s exact
    margin-only-vs-full-notional decision (same classify_symbol/
    max_leverage inputs) so the two can never diverge again. Short
    positions and futures/forex/crypto tickers are unaffected -- they were
    already correctly "cfd" via the existing rules.
    """
    margin_config = load_margin_config(MARGIN_CONFIG_PATH)

    # Unleveraged (or margin_config omitted): legacy behavior unchanged.
    assert kairos_papertrade.map_instrument_type({"ticker": "AAPL", "direction": "long"}) == "stock"
    assert kairos_papertrade.map_instrument_type(
        {"ticker": "AAPL", "direction": "long"}, margin_config, max_leverage=1.0,
    ) == "stock"

    # Leveraged: AAPL (equity_cfd, 20% margin < 100%) is now "cfd".
    assert kairos_papertrade.map_instrument_type(
        {"ticker": "AAPL", "direction": "long"}, margin_config, max_leverage=5.0,
    ) == "cfd"

    # Crypto stays "cfd" either way (already matched via the suffix rule,
    # unaffected by margin_config) -- but its class is 100% margin
    # (unleveraged), so classify_symbol alone would NOT have routed it to
    # "cfd" without the pre-existing suffix rule; confirms the two rules
    # compose correctly rather than one overriding the other.
    assert kairos_papertrade.map_instrument_type(
        {"ticker": "BTC-USD", "direction": "long"}, margin_config, max_leverage=5.0,
    ) == "cfd"

    # Short stays "cfd" regardless of margin_config/leverage (pre-existing rule).
    assert kairos_papertrade.map_instrument_type(
        {"ticker": "AAPL", "direction": "short"}, margin_config, max_leverage=1.0,
    ) == "cfd"
