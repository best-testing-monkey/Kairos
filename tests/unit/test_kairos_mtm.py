"""tests/unit/test_kairos_mtm.py — Unit tests for mark-to-market snapshots."""

import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest  # noqa: E402

from kairos_margin import load_margin_config  # noqa: E402
from kairos_mtm import compute_daily_snapshot, DailySnapshot, OpenPositionView, unrealized_pnl  # noqa: E402


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
