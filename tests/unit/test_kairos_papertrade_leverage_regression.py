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


def _candidate(ticker, entry, stop, target, expected_value=None):
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
    """
    ev = expected_value if expected_value is not None else entry * 0.05
    return (
        _stats_row(ticker, "LONG", entry, stop, target, ev),
        _advice_row(entry, ev, ticker=ticker),
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
    TICK1/TICK2 open on day1 UNCHECKED (admission_check is skipped whenever
    `last_snapshot` is None -- true for the very first order batch of a
    run, see kairos_papertrade.py's `_place_order_if_admitted` docstring),
    establishing a real MTM snapshot. TICK3..TICK6 are then offered as a
    4-candidate batch on day2, admitted/rejected one-by-one against that
    snapshot via `_place_batch_orders`' running initial-margin-used total
    and the (very tight, 0.1) margin_utilization_cap -- expected to admit
    only the first of the four (proving the cap is actually load-bearing
    here, not vacuously satisfied), which the test verifies via
    margin_rejected_count > 0.
    """
    day0 = datetime(2024, 1, 1)
    day1 = datetime(2024, 1, 2)
    day2 = datetime(2024, 1, 3)

    wave1 = ["TICK1", "TICK2"]
    wave2 = ["TICK3", "TICK4", "TICK5", "TICK6"]
    all_tickers = wave1 + wave2

    dated_rows = _dated_rows([
        (day0, [_candidate(t, 100.0, 90.0, 140.0) for t in wave1]),
        (day1, [_candidate(t, 100.0, 90.0, 140.0) for t in wave2]),
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
            "--max-leverage", "2.0", "--margin-utilization", "0.1",
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

        bound = peak_equity * 2.0 * 0.1 / min_initial_margin_pct_fraction
        assert peak_gross_notional <= bound, (
            f"peak gross_notional={peak_gross_notional} on {peak_date} exceeds bound={bound} "
            f"(equity={peak_equity} * max_leverage=2.0 * margin_utilization_cap=0.1 / "
            f"min_initial_margin_pct={min_initial_margin_pct_fraction})"
        )

        # The cap must actually have been load-bearing for this to be a
        # meaningful test (not just a vacuously-true inequality): at least
        # one of the wave-2 candidates should have been rejected.
        assert meta["margin_rejected_count"] > 0
        assert meta["margin_rejected_count"] < len(wave2)  # not ALL rejected either
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
