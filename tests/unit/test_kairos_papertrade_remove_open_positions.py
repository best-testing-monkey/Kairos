"""Validation tests for kairos_papertrade.remove_all_open_positions (Fix 1 of the
papertrade loss investigation -- see docs/papertrade_loss_analysis.md).

Requires a live `phantom` install (the `phantom-ledger` package) but no GPU/model
download and no network access: orders are "filled" directly via phantom's own
`OrderManager.handle_fill` rather than through a real price-data-driven backtest, so
these tests are fast and deterministic.

Unlike tests/unit/test_kairos_papertrade_loss_repro.py (which pins numbers from a real
historical run and is off-limits to edit), this file is a fresh, from-scratch
validation of the NEW remove_all_open_positions function and is fine to extend/edit
going forward.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

ph = pytest.importorskip("phantom", reason="phantom_ledger not installed")

from datetime import datetime, timezone  # noqa: E402

from phantom.costs.engine import CostEngine  # noqa: E402
from phantom.db.repositories.account_repo import AccountRepo  # noqa: E402
from phantom.db.repositories.order_repo import OrderRepo  # noqa: E402
from phantom.db.repositories.position_repo import PositionRepo  # noqa: E402
from phantom.engine.order_manager import OrderManager  # noqa: E402
from phantom.models.order import Order  # noqa: E402

from kairos_papertrade import remove_all_open_positions  # noqa: E402


CAPITAL = 200.0


@pytest.fixture
def client(tmp_path):
    os.environ["PHANTOM_DATA"] = str(tmp_path)
    c = ph.Phantom(data_dir=str(tmp_path))
    import phantom.profiles as _profiles_pkg
    profile_path = os.path.join(os.path.dirname(_profiles_pkg.__file__), "ibkr.json")
    c.brokers.load(profile_path)
    try:
        yield c
    finally:
        # phantom.Phantom has no public close()/context-manager; close the
        # sqlite3 connection on its private `_conn` (same attribute
        # kairos_papertrade.py itself already reaches into) so it doesn't
        # leak until an unrelated later test's GC cycle finalizes it -- see
        # the matching fixture in test_kairos_papertrade_loss_repro.py.
        c._conn.close()


def _open_position(client, account, ticker, direction, entry_price, quantity):
    """Fill an order directly via phantom's own OrderManager.handle_fill --
    exactly the code path a real backtest uses (verified against
    phantom/engine/order_manager.py) -- without needing real price-bar data or
    network access. Returns the resulting open Position.
    """
    order_repo = OrderRepo(client._conn)
    account_repo = AccountRepo(client._conn)
    position_repo = PositionRepo(client._conn)
    profile = client.brokers.get("IBKR")
    cost_engine = CostEngine(profile)
    manager = OrderManager(order_repo, account_repo, cost_engine)

    order = Order(
        account_id=account.id, ticker=ticker, instrument_type="cfd",
        direction=direction, order_type="market", quantity=quantity,
        take_profit=None, stop_loss=None,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    placed = manager.place(account.id, order)
    filled = placed.model_copy(update={
        "fill_price": entry_price,
        "filled_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    })
    _, position = manager.handle_fill(filled, position_repo)
    return position


class TestRemoveAllOpenPositions:
    def test_refunds_cash_and_removes_position(self, client):
        account = client.accounts.create(
            name="acct1", account_type="algorithm", broker="IBKR",
            capital=CAPITAL, currency="EUR", algorithm_id="test", algorithm_version="1d",
        )
        cash_before_open = client.accounts.get(account.id).cash
        assert cash_before_open == pytest.approx(CAPITAL)

        pos = _open_position(client, account, "BTC-USD", "long", 50000.0, 0.001)
        cash_after_open = client.accounts.get(account.id).cash
        assert cash_after_open < cash_before_open  # entry deduction actually happened

        open_before = client.positions.list(account_name="acct1", status="open")
        assert len(open_before) == 1

        remove_all_open_positions(client, account.id, "acct1")

        cash_after_remove = client.accounts.get(account.id).cash
        assert cash_after_remove == pytest.approx(cash_before_open, abs=1e-9)

        open_after = client.positions.list(account_name="acct1", status="open")
        closed_after = client.positions.list(account_name="acct1", status="closed")
        assert open_after == []
        assert closed_after == []

    def test_refund_matches_exact_entry_deduction_formula(self, client):
        """Refund must equal entry_price*quantity + the four entry-side cost
        fields phantom's own OrderManager.handle_fill deducted -- not just the
        bare notional -- so cash ends up EXACTLY where it started, not off by
        the entry costs."""
        account = client.accounts.create(
            name="acct2", account_type="algorithm", broker="IBKR",
            capital=CAPITAL, currency="EUR", algorithm_id="test", algorithm_version="1d",
        )
        cash_before_open = client.accounts.get(account.id).cash

        pos = _open_position(client, account, "ETH-USD", "short", 3000.0, 0.01)
        expected_refund = (
            pos.entry_price * pos.quantity
            + pos.commission_entry + pos.spread_cost
            + pos.slippage_cost + pos.fx_conversion_cost
        )
        assert expected_refund > pos.entry_price * pos.quantity  # costs are non-zero (fx, at least)

        remove_all_open_positions(client, account.id, "acct2")

        cash_after_remove = client.accounts.get(account.id).cash
        assert cash_after_remove == pytest.approx(cash_before_open, abs=1e-9)

    def test_multiple_open_positions_all_removed(self, client):
        account = client.accounts.create(
            name="acct3", account_type="algorithm", broker="IBKR",
            capital=CAPITAL, currency="EUR", algorithm_id="test", algorithm_version="1d",
        )
        cash_before_open = client.accounts.get(account.id).cash

        _open_position(client, account, "BTC-USD", "long", 50000.0, 0.001)
        _open_position(client, account, "ETH-USD", "short", 3000.0, 0.005)

        assert len(client.positions.list(account_name="acct3", status="open")) == 2

        remove_all_open_positions(client, account.id, "acct3")

        assert client.positions.list(account_name="acct3", status="open") == []
        cash_after_remove = client.accounts.get(account.id).cash
        assert cash_after_remove == pytest.approx(cash_before_open, abs=1e-9)

    def test_closed_positions_are_untouched(self, client):
        """remove_all_open_positions must only touch status='open' rows --
        an already-closed position (a real win/loss) must survive unchanged."""
        account = client.accounts.create(
            name="acct4", account_type="algorithm", broker="IBKR",
            capital=CAPITAL, currency="EUR", algorithm_id="test", algorithm_version="1d",
        )
        open_pos = _open_position(client, account, "BTC-USD", "long", 50000.0, 0.001)
        closed_pos = client.positions.close(
            open_pos.id, close_reason="tp", exit_price=51000.0,
            exit_datetime=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        cash_after_close = client.accounts.get(account.id).cash

        remove_all_open_positions(client, account.id, "acct4")

        assert client.positions.list(account_name="acct4", status="open") == []
        closed = client.positions.list(account_name="acct4", status="closed")
        assert len(closed) == 1
        assert closed[0].id == closed_pos.id
        assert closed[0].realized_pnl == closed_pos.realized_pnl
        # cash must be unaffected by removal since there was nothing open
        assert client.accounts.get(account.id).cash == pytest.approx(cash_after_close, abs=1e-9)

    def test_noop_when_nothing_open(self, client):
        account = client.accounts.create(
            name="acct5", account_type="algorithm", broker="IBKR",
            capital=CAPITAL, currency="EUR", algorithm_id="test", algorithm_version="1d",
        )
        cash_before = client.accounts.get(account.id).cash
        remove_all_open_positions(client, account.id, "acct5")  # must not raise
        assert client.accounts.get(account.id).cash == pytest.approx(cash_before)
