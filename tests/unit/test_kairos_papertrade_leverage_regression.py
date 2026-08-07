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
