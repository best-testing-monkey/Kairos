"""E5-S14 -- Frozen-fixture MTM repro test.

Replays the already-recorded CLOSED positions from the frozen 2026-07-26
fixture (`tests/data/kairos_papertrade_20260726_phantom.db`, the same fixture
`test_kairos_papertrade_loss_repro.py` pins) through the MTM machinery
(`kairos_mtm.compute_daily_snapshot`/`compute_daily_financing_total` +
`kairos_papertrade._insert_mtm_daily_row`), then calls `compute_final_metrics`
on the result.

This fixture predates the MTM feature (E4-S08..E4-S13): it has no
`kairos_mtm_daily` rows of its own, and there is no cached daily-OHLC price
fixture in this repo for this window's tickers, so a live day-by-day mark is
not possible without network access (forbidden in this test suite). This test
therefore builds its OWN coarser replay directly from the positions' stored
`entry_price`/`exit_price`/`entry_datetime`/`exit_datetime` (real historical
data, not synthetic) rather than a live phantom day loop:

- "Trading days in the window" (`D`) is defined as the ordered, deduplicated
  union of every closed position's `entry_datetime.date()` and
  `exit_datetime.date()`. This is a deliberate, honest stand-in for a true
  calendar of trading days -- it's the only set of dates this frozen fixture
  can tell us about without re-running report generation.
- A still-open (not yet exited) position on a given day in `D` has no real
  interim mark available, so it is marked at its OWN entry_price (i.e. zero
  unrealized P&L until its actual, known exit day). This is a deliberate
  simplification -- it means this replay's MTM curve, like the closed-trade
  curve it's compared against, does NOT capture true intra-trade unrealized
  swings. See test_drawdown_comparison's docstring below for the direct
  consequence of this choice on the drawdown-ordering assertion.
- Cash deltas (`_fill_cash_delta`/`_close_cash_delta`) are applied on a
  position's own entry/exit day exactly once each, per the ticket brief. The
  daily MTM *mark* (the set of positions fed into `compute_daily_snapshot`),
  however, mirrors `main()`'s real day loop: it only includes positions still
  open GOING INTO the next day (`current_open`, which by construction
  excludes anything that closed on the current day, since phantom's own
  `runner.backtest()` already transitioned it out of `status='open'` before
  `current_open` is fetched). A position that exits on day `d` is therefore
  excluded from day `d`'s mark entirely -- its full economic effect is
  already captured by `_close_cash_delta` on that same day, and marking it
  again at `exit_price` would double-count its gross P&L on top of the
  already-cost-corrected cash delta (verified empirically while building this
  test: including same-day exits in the mark broke the final-equity
  convergence identity below by exactly the sum of double-counted gross P&L;
  excluding them, matching `main()`'s real semantics, makes it converge to
  machine precision).

Like its sibling `test_kairos_papertrade_loss_repro.py`, this is a
CHARACTERIZATION test: it pins what THIS replay construction actually
produces on real historical data, not a claim that any particular P&L or
drawdown number is "correct" in isolation.
"""
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

FIXTURE_DB_V2_FIXED = Path(__file__).parent.parent / "data" / "kairos_papertrade_20260726_phantom.db"
ACCOUNT_NAME_V2_FIXED = "kairos_papertrade_202607261257"

CAPITAL = 200.0

MARGIN_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "margin_ibkr.yaml"
MAX_LEVERAGE = 1.0  # this fixture is a legacy/no-margin run


def _phantom_client(tmp_path, fixture_db):
    if not fixture_db.exists():
        pytest.skip(f"fixture DB missing: {fixture_db}")
    shutil.copy(fixture_db, tmp_path / "phantom.db")
    import phantom as ph
    client = ph.Phantom(data_dir=str(tmp_path))
    try:
        yield client
    finally:
        # See test_kairos_papertrade_loss_repro.py's _phantom_client for why this
        # explicit close is needed (phantom.Phantom exposes no close()/context
        # manager; without it a later, unrelated test sees a stray
        # ResourceWarning: unclosed database).
        client._conn.close()


@pytest.fixture
def phantom_client(tmp_path):
    """A phantom.Phantom client on a throwaway COPY of the frozen 2026-07-26
    fixture -- never mutate the checked-in DB itself."""
    yield from _phantom_client(tmp_path, FIXTURE_DB_V2_FIXED)


def _replay_mtm(client, account_name, capital, margin_config):
    """Walk the trading-day window D (union of entry/exit dates of every
    closed position) and populate kairos_mtm_daily by reusing the existing,
    already-unit-tested helpers -- see module docstring for the replay
    definition/simplifications. Returns (corrected_cash, trading_days)."""
    from kairos_papertrade import (
        _ensure_mtm_daily_table, _insert_mtm_daily_row, _fill_cash_delta,
        _close_cash_delta, _use_full_notional,
    )
    from kairos_mtm import OpenPositionView, compute_daily_snapshot, compute_daily_financing_total

    _ensure_mtm_daily_table(client._conn)

    closed = client.positions.list(account_name=account_name, status="closed")

    entry_dates = {pos.entry_datetime.date() for pos in closed}
    exit_dates = {pos.exit_datetime.date() for pos in closed}
    trading_days = sorted(entry_dates | exit_dates)

    corrected_cash = capital
    financing_accrued_total = 0.0
    filled_ids: set = set()
    closed_ids: set = set()

    for d in trading_days:
        open_today = [pos for pos in closed if pos.entry_datetime.date() <= d <= pos.exit_datetime.date()]

        for pos in open_today:
            if pos.id not in filled_ids:
                corrected_cash += _fill_cash_delta(
                    pos, include_notional=_use_full_notional(pos.ticker, margin_config, MAX_LEVERAGE),
                )
                filled_ids.add(pos.id)
            if pos.exit_datetime.date() == d and pos.id not in closed_ids:
                corrected_cash += _close_cash_delta(
                    pos, include_notional=_use_full_notional(pos.ticker, margin_config, MAX_LEVERAGE),
                )
                closed_ids.add(pos.id)

        # MTM mark: only positions still open GOING INTO the next day (excludes
        # today's exits) -- see module docstring for why this must diverge from
        # the raw "open_today" set used for cash-delta bookkeeping above.
        mark_today = [pos for pos in open_today if pos.exit_datetime.date() != d]
        mtm_positions = [
            OpenPositionView(
                ticker=pos.ticker, direction=pos.direction, entry_price=pos.entry_price,
                quantity=pos.quantity,
                entry_costs=(
                    pos.commission_entry + pos.spread_cost + pos.slippage_cost + pos.fx_conversion_cost
                ),
            )
            for pos in mark_today
        ]
        bars_by_ticker = {pos.ticker: {"date": d, "close": pos.entry_price} for pos in mark_today}

        financing_day = (
            compute_daily_financing_total(mtm_positions, bars_by_ticker, margin_config)
            if mtm_positions else 0.0
        )
        corrected_cash -= financing_day
        financing_accrued_total += financing_day

        if mtm_positions:
            snapshot = replace(
                compute_daily_snapshot(mtm_positions, bars_by_ticker, corrected_cash, margin_config),
                financing_accrued_day=financing_day,
            )
        else:
            from kairos_mtm import DailySnapshot
            snapshot = DailySnapshot(
                date=d, cash=corrected_cash, unrealized_pnl=0.0, equity=corrected_cash,
                gross_notional=0.0, initial_margin_used=0.0, maintenance_margin_used=0.0,
                free_margin=corrected_cash, margin_utilization=0.0,
                financing_accrued_day=financing_day, liquidations=0,
            )

        _insert_mtm_daily_row(client._conn, account_name, snapshot, financing_accrued_total)

    return corrected_cash, trading_days


class TestFrozenFixtureMtmRepro:
    def test_row_count_equals_trading_days(self, phantom_client):
        from kairos_margin import load_margin_config

        margin_config = load_margin_config(MARGIN_CONFIG_PATH)
        _, trading_days = _replay_mtm(phantom_client, ACCOUNT_NAME_V2_FIXED, CAPITAL, margin_config)

        (row_count,) = phantom_client._conn.execute(
            "SELECT count(*) FROM kairos_mtm_daily WHERE account_name = ?", (ACCOUNT_NAME_V2_FIXED,)
        ).fetchone()
        assert row_count == len(trading_days)

    def test_final_mtm_equity_equals_final_closed_trade_equity(self, phantom_client):
        """MATHEMATICALLY GUARANTEED by construction (see module docstring +
        _fill_cash_delta/_close_cash_delta docstrings in kairos_papertrade.py):
        every position gets exactly one fill delta and exactly one close delta
        applied over the whole replay, so final corrected_cash ==
        CAPITAL + sum(compute_corrected_realized_pnl(pos) for all closed
        positions) -- exactly build_closed_trade_equity_curve's own final
        equity point. On the last trading day no position remains open (every
        position has exited by construction of D), so the final MTM
        DailySnapshot.equity == corrected_cash (zero unrealized P&L). If this
        assertion ever fails, that means the replay loop has a bug (e.g. a
        delta applied twice or on the wrong date), not a legitimate finding --
        see the ticket brief for this reasoning.
        """
        from kairos_margin import load_margin_config
        from kairos_papertrade import compute_final_metrics, build_closed_trade_equity_curve

        margin_config = load_margin_config(MARGIN_CONFIG_PATH)
        account = phantom_client.accounts.get(ACCOUNT_NAME_V2_FIXED)
        corrected_cash, _trading_days = _replay_mtm(
            phantom_client, ACCOUNT_NAME_V2_FIXED, CAPITAL, margin_config,
        )

        closed = phantom_client.positions.list(account_name=ACCOUNT_NAME_V2_FIXED, status="closed")
        closed_trade_curve = build_closed_trade_equity_curve(closed, CAPITAL)
        final_closed_trade_equity = closed_trade_curve[-1].equity

        (final_mtm_equity,) = phantom_client._conn.execute(
            "SELECT equity FROM kairos_mtm_daily WHERE account_name = ? ORDER BY date DESC LIMIT 1",
            (ACCOUNT_NAME_V2_FIXED,),
        ).fetchone()

        assert corrected_cash == pytest.approx(final_closed_trade_equity, rel=1e-9)
        assert final_mtm_equity == pytest.approx(final_closed_trade_equity, rel=1e-9)

        start_dt = min(pos.entry_datetime for pos in closed)
        metrics = compute_final_metrics(
            phantom_client, account.id, ACCOUNT_NAME_V2_FIXED, CAPITAL, start_dt=start_dt,
        )
        assert metrics["mtm_total_return_pct"] == pytest.approx(metrics["pct_profit"], rel=1e-9)

    def test_drawdown_comparison(self, phantom_client):
        """The ticket asks to assert mtm_max_drawdown_pct >= pct_max_drawdown
        (the "documented understatement" of the closed-trade step-function
        curve vs. a true MTM curve, per DESIGN_DOC_mtm_margin_leverage.md
        Section 6.2). That inequality's usual justification is that the MTM
        curve captures true intra-trade unrealized swings the closed-trade
        curve misses. THIS replay's "mark still-open positions at entry_price"
        simplification (see module docstring) means our MTM curve has that
        SAME blind spot -- so the inequality is NOT guaranteed to hold here by
        construction, unlike the equity-convergence test above. We compute
        both values first and check empirically rather than assuming: on THIS
        fixture it DOES hold (see the pinned values below), driven by
        date-bucketed cash depletion while many positions are concurrently
        filled but not yet closed, not by captured price risk. Both values are
        asserted sane (non-negative) unconditionally, and the pinned
        equality/inequality below is a characterization pin for this specific
        dataset and replay construction -- not a general proof.
        """
        from kairos_margin import load_margin_config
        from kairos_papertrade import compute_final_metrics

        margin_config = load_margin_config(MARGIN_CONFIG_PATH)
        account = phantom_client.accounts.get(ACCOUNT_NAME_V2_FIXED)
        _replay_mtm(phantom_client, ACCOUNT_NAME_V2_FIXED, CAPITAL, margin_config)

        closed = phantom_client.positions.list(account_name=ACCOUNT_NAME_V2_FIXED, status="closed")
        start_dt = min(pos.entry_datetime for pos in closed)
        metrics = compute_final_metrics(
            phantom_client, account.id, ACCOUNT_NAME_V2_FIXED, CAPITAL, start_dt=start_dt,
        )

        mtm_dd = metrics["mtm_max_drawdown_pct"]
        closed_dd = metrics["pct_max_drawdown"]

        assert mtm_dd >= 0.0
        assert closed_dd >= 0.0

        # Observed on this specific fixture with this specific replay
        # construction (pinned the same way EXPECTED_METRICS_V2 pins values in
        # test_kairos_papertrade_loss_repro.py): the inequality DOES hold here
        # (mtm_max_drawdown_pct ~58.3% vs pct_max_drawdown ~9.2%) -- driven
        # mostly by this replay's date-bucketed cash swings while many
        # positions are concurrently filled but not yet closed, not by
        # captured intra-trade price risk (this replay's interim marks are
        # flat at entry_price, per the module docstring). This is a
        # characterization pin for THIS dataset/construction, not a proof
        # that mtm_max_drawdown_pct >= pct_max_drawdown holds for any run --
        # a replay with a true daily price-bar fixture could show the
        # opposite ordering in principle (finer-resolution per-trade closes
        # vs. coarser date-bucketed MTM points).
        assert mtm_dd == pytest.approx(58.33185585472338, rel=1e-6)
        assert closed_dd == pytest.approx(9.209712710925718, rel=1e-9)
        assert mtm_dd >= closed_dd
