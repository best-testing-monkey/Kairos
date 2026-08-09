"""tests/unit/test_kairos_mtm.py — Unit tests for mark-to-market snapshots."""

import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest  # noqa: E402

from ..allocation import AllocationConfig  # noqa: E402
from kairos_margin import load_margin_config  # noqa: E402
from kairos_mtm import (  # noqa: E402
    admission_check,
    compute_daily_financing_total,
    compute_daily_snapshot,
    daily_financing,
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
    assert tickers[0] == "AAPL"


def test_liquidation_check_post_invariant_holds(cfg) -> None:
    """Post-liquidation, equity >= closeout_fraction * initial_margin_used_post."""
    # AAPL: notional 5000, IM 20% = 1000, MM 10% = 500.
    # MSFT: notional 10000, IM 20% = 2000, MM 10% = 1000.
    # Total IM = 3000, equity = 1000 => ratio = 0.333 < 0.5 => liquidate.
    # Greedy order liquidates MSFT first (largest MM release: 1000 > 500).
    # After removing MSFT: IM = 3000 - 2000 = 1000; equity(1000) >= 0.5*1000 => safe, stop.
    positions = [
        OpenPositionView(
            ticker="AAPL", direction="long", entry_price=100.0, quantity=50.0, entry_costs=0.0
        ),
        OpenPositionView(
            ticker="MSFT", direction="long", entry_price=200.0, quantity=50.0, entry_costs=0.0
        ),
    ]
    snapshot = _make_snapshot(equity=1000.0, initial_margin_used=3000.0)
    tickers, post_eq, ruined = liquidation_check(snapshot, positions, cfg)

    assert tickers == ["MSFT"]
    assert post_eq >= 0.5 * (snapshot.initial_margin_used - 2000.0)
    assert ruined is False


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


def test_daily_financing_long_exact_value(cfg) -> None:
    """Long position financing: notional * (benchmark + spread) / 360."""
    # AAPL (equity CFD): entry_price=100, qty=10 => notional=1000
    # Benchmark=3.15%, Spread=1.5% => (3.15+1.5) / 360 = 4.65 / 360 ≈ 0.01291667
    # Expected: 1000 * 0.01291667 ≈ 12.91667
    pos = OpenPositionView(
        ticker="AAPL", direction="long", entry_price=100.0, quantity=10.0, entry_costs=0.0
    )
    close_price = 100.0
    margin_class = cfg.classes["equity_cfd"]
    result = daily_financing(pos, close_price, margin_class, cfg)

    expected = 1000.0 * (3.15 + 1.5) / 360.0
    assert abs(result - expected) < 1e-9


def test_daily_financing_short_with_borrow_exact_value(cfg) -> None:
    """Short position financing: borrow_cost - financing_credit_if_positive."""
    # AAPL short: entry_price=100, qty=10 => notional=1000
    # Borrow cost: 1000 * 1.0 / 360 ≈ 2.77778
    # Financing credit: max(0, 1000 * (3.15 - 1.5) / 360) = 1000 * 1.65 / 360 ≈ 4.58333
    # Net cost: 2.77778 - 4.58333 ≈ -1.80556 (credit to account)
    pos = OpenPositionView(
        ticker="AAPL", direction="short", entry_price=100.0, quantity=10.0, entry_costs=0.0
    )
    close_price = 100.0
    margin_class = cfg.classes["equity_cfd"]
    result = daily_financing(pos, close_price, margin_class, cfg)

    borrow_cost = 1000.0 * 1.0 / 360.0
    financing_credit = max(0.0, 1000.0 * (3.15 - 1.5) / 360.0)
    expected = borrow_cost - financing_credit
    assert abs(result - expected) < 1e-9


def test_daily_financing_spot_returns_zero(cfg) -> None:
    """Spot classes (initial_margin_pct == 100) return 0.0."""
    # BTC-USD is crypto_spot: initial_margin_pct=100.0
    pos = OpenPositionView(
        ticker="BTC-USD", direction="long", entry_price=40000.0, quantity=0.1, entry_costs=0.0
    )
    close_price = 41000.0
    margin_class = cfg.classes["crypto_spot"]
    result = daily_financing(pos, close_price, margin_class, cfg)

    assert result == 0.0


def test_daily_financing_benchmark_and_spread_zero(cfg) -> None:
    """Edge case: zero spread (credit rate approaches benchmark)."""
    # Create a margin class with zero spread
    # For long: should return benchmark (no spread added)
    # For short: credit rate = benchmark - 0 = benchmark (if positive)
    from kairos_margin import MarginClass  # noqa: E402

    zero_spread_class = MarginClass(
        name="test_zero",
        initial_margin_pct=20.0,
        maintenance_margin_pct=10.0,
        financing_spread_pct=0.0,
    )

    # Long position: 1000 * (3.15 + 0) / 360 = 8.75
    pos_long = OpenPositionView(
        ticker="TEST", direction="long", entry_price=100.0, quantity=10.0, entry_costs=0.0
    )
    result_long = daily_financing(pos_long, 100.0, zero_spread_class, cfg)
    expected_long = 1000.0 * (3.15 + 0.0) / 360.0
    assert abs(result_long - expected_long) < 1e-9

    # Short position: 1000 * 1.0 / 360 - max(0, 1000 * (3.15 - 0) / 360)
    # = 2.77778 - 8.75 = -5.97222 (credit to account)
    pos_short = OpenPositionView(
        ticker="TEST", direction="short", entry_price=100.0, quantity=10.0, entry_costs=0.0
    )
    result_short = daily_financing(pos_short, 100.0, zero_spread_class, cfg)
    borrow_cost = 1000.0 * 1.0 / 360.0
    financing_credit = max(0.0, 1000.0 * (3.15 - 0.0) / 360.0)
    expected_short = borrow_cost - financing_credit
    assert abs(result_short - expected_short) < 1e-9


def test_compute_daily_financing_total_multiple_positions(cfg) -> None:
    """Sum financing across multiple mixed positions."""
    positions = [
        OpenPositionView(
            ticker="AAPL", direction="long", entry_price=100.0, quantity=10.0, entry_costs=0.0
        ),
        OpenPositionView(
            ticker="EURUSD=X", direction="short", entry_price=1.1, quantity=100000.0, entry_costs=0.0
        ),
        OpenPositionView(
            ticker="BTC-USD", direction="long", entry_price=40000.0, quantity=0.1, entry_costs=0.0
        ),
    ]
    bars_by_ticker = {
        "AAPL": {"date": datetime.date(2026, 8, 7), "close": 100.0},
        "EURUSD=X": {"date": datetime.date(2026, 8, 7), "close": 1.1},
        "BTC-USD": {"date": datetime.date(2026, 8, 7), "close": 41000.0},
    }

    total = compute_daily_financing_total(positions, bars_by_ticker, cfg)

    # Compute expected values:
    # AAPL long: 1000 * (3.15 + 1.5) / 360 = 12.91667
    aapl_fin = 1000.0 * (3.15 + 1.5) / 360.0

    # EURUSD=X short: notional=110000
    # Borrow: 110000 * 1.0 / 360 ≈ 305.55556
    # Credit: max(0, 110000 * (3.15 - 1.5) / 360) = 110000 * 1.65 / 360 ≈ 505.20833
    # Net: 305.55556 - 505.20833 ≈ -199.65278
    eurusd_borrow = 110000.0 * 1.0 / 360.0
    eurusd_credit = max(0.0, 110000.0 * (3.15 - 1.5) / 360.0)
    eurusd_fin = eurusd_borrow - eurusd_credit

    # BTC-USD spot: 0
    btc_fin = 0.0

    expected_total = aapl_fin + eurusd_fin + btc_fin

    assert abs(total - expected_total) < 1e-6
