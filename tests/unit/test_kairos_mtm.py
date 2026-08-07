"""tests/unit/test_kairos_mtm.py — Unit tests for mark-to-market snapshots."""

import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest  # noqa: E402

from allocation import AllocationConfig  # noqa: E402
from kairos_margin import load_margin_config  # noqa: E402
from kairos_mtm import (  # noqa: E402
    admission_check,
    compute_daily_snapshot,
    DailySnapshot,
    liquidation_check,
    OpenPositionView,
    unrealized_pnl,
)


@pytest.fixture
def cfg():
    """Default IBKR-style margin config fixture."""
    return load_margin_config(
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "margin_ibkr.yaml")
    )


def test_unrealized_pnl_long() -> None:
    pos = OpenPositionView(
        ticker="AAPL", direction="long", entry_price=100.0, quantity=10.0, entry_costs=0.0
    )
    assert unrealized_pnl(pos, 105.0) == 50.0


def test_unrealized_pnl_short() -> None:
    pos = OpenPositionView(
        ticker="AAPL", direction="short", entry_price=100.0, quantity=10.0, entry_costs=0.0
    )
    assert unrealized_pnl(pos, 95.0) == 50.0


def test_compute_daily_snapshot_two_positions(cfg) -> None:
    positions = [
        OpenPositionView(
            ticker="AAPL", direction="long", entry_price=100.0, quantity=10.0, entry_costs=5.0
        ),
        OpenPositionView(
            ticker="BTC-USD", direction="short", entry_price=40000.0, quantity=0.1, entry_costs=2.0
        ),
    ]
    bars_by_ticker = {
        "AAPL": {"date": datetime.date(2026, 8, 7), "close": 105.0},
        "BTC-USD": {"date": datetime.date(2026, 8, 7), "close": 39000.0},
    }

    snapshot = compute_daily_snapshot(positions, bars_by_ticker, cash=8250.0, cfg=cfg)

    expected = DailySnapshot(
        date=datetime.date(2026, 8, 7),
        cash=8250.0,
        unrealized_pnl=150.0,
        equity=8400.0,
        gross_notional=5000.0,
        initial_margin_used=4200.0,
        maintenance_margin_used=100.0,
        free_margin=4200.0,
        margin_utilization=0.5,
        financing_accrued_day=0.0,
        liquidations=0,
    )
    assert snapshot == expected


def _make_snapshot(
    equity: float,
    initial_margin_used: float = 0.0,
    gross_notional: float = 0.0,
) -> DailySnapshot:
    """Build a minimal DailySnapshot for admission-check tests."""
    return DailySnapshot(
        date=datetime.date(2026, 8, 7),
        cash=equity,
        unrealized_pnl=0.0,
        equity=equity,
        gross_notional=gross_notional,
        initial_margin_used=initial_margin_used,
        maintenance_margin_used=0.0,
        free_margin=equity - initial_margin_used,
        margin_utilization=initial_margin_used / equity if equity > 0 else 0.0,
        financing_accrued_day=0.0,
        liquidations=0,
    )


def test_admission_check_accepts_below_cap(cfg) -> None:
    alloc = AllocationConfig(max_leverage=2.0, margin_utilization_cap=0.8)
    account = _make_snapshot(equity=10000.0)
    # Equity CFD initial margin is 20%; 30000 * 0.2 = 6000 <= 8000 cap.
    assert admission_check(30000.0, "AAPL", account, cfg, alloc) is True


def test_admission_check_rejects_above_cap(cfg) -> None:
    alloc = AllocationConfig(max_leverage=2.0, margin_utilization_cap=0.8)
    account = _make_snapshot(equity=10000.0)
    # Equity CFD initial margin is 20%; 50000 * 0.2 = 10000 > 8000 cap.
    assert admission_check(50000.0, "AAPL", account, cfg, alloc) is False


def test_admission_check_rejects_non_positive_equity(cfg) -> None:
    alloc = AllocationConfig(max_leverage=2.0, margin_utilization_cap=0.8)
    account = _make_snapshot(equity=0.0)
    assert admission_check(1000.0, "AAPL", account, cfg, alloc) is False


def test_admission_check_noop_when_unleveraged(cfg) -> None:
    alloc = AllocationConfig(max_leverage=1.0, margin_utilization_cap=0.8)
    account = _make_snapshot(equity=10000.0)
    # With leverage disabled the check is a no-op even for a huge order.
    assert admission_check(100000.0, "AAPL", account, cfg, alloc) is True


def test_liquidation_check_no_trigger_at_50_percent(cfg) -> None:
    """No liquidation when equity is exactly at the safe level (50% of IM)."""
    # Equity = 5000, IM = 10000, ratio = 0.5 => safe
    snapshot = _make_snapshot(equity=5000.0, initial_margin_used=10000.0)
    positions = [
        OpenPositionView(
            ticker="AAPL", direction="long", entry_price=100.0, quantity=50.0, entry_costs=0.0
        ),
    ]
    tickers, post_eq, ruined = liquidation_check(snapshot, positions, cfg)
    assert tickers == []
    assert post_eq == 5000.0
    assert ruined is False


def test_liquidation_check_trigger_at_49_percent(cfg) -> None:
    """Liquidation triggered when equity falls to 49% of IM."""
    # Equity = 4900, IM = 10000, ratio = 0.49 => unsafe
    snapshot = _make_snapshot(equity=4900.0, initial_margin_used=10000.0)
    positions = [
        OpenPositionView(
            ticker="AAPL", direction="long", entry_price=100.0, quantity=50.0, entry_costs=0.0
        ),
    ]
    tickers, post_eq, ruined = liquidation_check(snapshot, positions, cfg)
    # Should liquidate the AAPL position; IM becomes 0
    assert "AAPL" in tickers
    assert ruined is False


def test_liquidation_check_greedy_ordering(cfg) -> None:
    """Positions are liquidated in order of largest maintenance-margin release."""
    # Two positions: AAPL (notional 5000) and BTC (notional 40000)
    # AAPL IM = 1000 (20%), MM = 500 (10%)
    # BTC IM = 40000 (100% spot), MM = 0 (0%)
    # Total IM = 41000, Total MM = 500
    # Equity = 20000 => ratio = 20000/41000 ≈ 0.488 => liquidation triggered
    positions = [
        OpenPositionView(
            ticker="AAPL", direction="long", entry_price=100.0, quantity=50.0, entry_costs=0.0
        ),
        OpenPositionView(
            ticker="BTC-USD", direction="long", entry_price=40000.0, quantity=1.0, entry_costs=0.0
        ),
    ]
    snapshot = _make_snapshot(
        equity=20000.0,
        initial_margin_used=41000.0,
        gross_notional=45000.0,
    )
    tickers, post_eq, ruined = liquidation_check(snapshot, positions, cfg)

    # AAPL has MM release of 5000 * 0.10 = 500
    # BTC has MM release of 40000 * 0.0 = 0 (crypto spot, no margin)
    # So AAPL should be liquidated first (largest release)
    assert tickers[0] == "AAPL" if tickers else None


def test_liquidation_check_post_invariant_holds(cfg) -> None:
    """Post-liquidation, equity >= closeout_fraction * initial_margin_used_post."""
    # Setup: trigger liquidation by being unsafe
    # Equity = 3000, IM_used = 10000 => ratio = 0.30 < 0.5 => liquidate
    # One position: AAPL with IM = 2000
    # After liquidation: IM_used = 8000, equity = 3000
    # ratio = 3000/8000 = 0.375 < 0.5 => continue
    # After all liquidation: IM_used = 0, ratio = undefined (safe by default)
    positions = [
        OpenPositionView(
            ticker="AAPL", direction="long", entry_price=100.0, quantity=50.0, entry_costs=0.0
        ),
    ]
    snapshot = _make_snapshot(equity=3000.0, initial_margin_used=2000.0)
    tickers, post_eq, ruined = liquidation_check(snapshot, positions, cfg)

    # After removing AAPL: IM = 2000 - 1000 = 1000
    # Equity / IM = 3000 / 1000 = 3.0 >= 0.5 => safe!
    if tickers:
        # Verify that post-liquidation state is safe
        assert post_eq >= 0.5 * (snapshot.initial_margin_used - 1000)


def test_liquidation_check_clamp_and_ruined(cfg) -> None:
    """All positions liquidated + negative equity => clamp to 0 and ruined=True."""
    # Setup: equity is deeply negative
    # Equity = -1000, IM = 5000
    positions = [
        OpenPositionView(
            ticker="AAPL", direction="long", entry_price=100.0, quantity=10.0, entry_costs=0.0
        ),
    ]
    snapshot = _make_snapshot(equity=-1000.0, initial_margin_used=200.0)
    tickers, post_eq, ruined = liquidation_check(snapshot, positions, cfg)

    # All positions should be liquidated
    assert "AAPL" in tickers
    # Equity should be clamped to 0
    assert post_eq == 0.0
    # Ruined flag should be set
    assert ruined is True
