"""_manage_positions: unset hold_days means no time-based cap (hold until
stop/target/end-of-data); explicit hold_days still expires as before."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from types import SimpleNamespace
import pandas as pd

from kairos_orchestrator import KairosOrchestrator
from kairos_backtest import Direction


def _bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def _fake_self():
    fake_self = SimpleNamespace(
        active_positions=[],
        capital=100_000.0,
        all_trades=[],
        tracker=SimpleNamespace(record_trade=lambda **kw: None),
        config=SimpleNamespace(fee_pct=0.0),
        equity_curve=[],
    )
    fake_self._calculate_pnl = lambda pos, exit_price: KairosOrchestrator._calculate_pnl(
        fake_self, pos, exit_price
    )
    return fake_self


def _position(hold_days_remaining):
    return {
        "symbol": "TEST",
        "direction": Direction.LONG,
        "size": 1.0,
        "entry_price": 100.0,
        "stop": 90.0,
        "target": 200.0,  # far away -- never triggers in these fixtures
        "strategy_name": "test_strat",
        "entry_date": pd.Timestamp("2026-01-01"),
        "hold_days_remaining": hold_days_remaining,
        "notional": 100.0,
        "time_exit_bar": None,
    }


def test_unset_hold_days_never_force_closes_on_a_quiet_bar():
    fake_self = _fake_self()
    fake_self.active_positions = [_position(None)]
    histories = {"TEST": pd.DataFrame([_bar(101, 102, 100, 101)], index=[pd.Timestamp("2026-01-02")])}

    KairosOrchestrator._manage_positions(fake_self, pd.Timestamp("2026-01-02"), histories)

    assert len(fake_self.active_positions) == 1  # still open
    assert fake_self.active_positions[0]["hold_days_remaining"] is None  # untouched, no countdown
    assert fake_self.all_trades == []


def test_explicit_hold_days_still_expires_as_before():
    fake_self = _fake_self()
    fake_self.active_positions = [_position(1)]
    histories = {"TEST": pd.DataFrame([_bar(101, 102, 100, 101)], index=[pd.Timestamp("2026-01-02")])}

    # day 1: quiet bar, decrements to 0, stays open
    KairosOrchestrator._manage_positions(fake_self, pd.Timestamp("2026-01-02"), histories)
    assert len(fake_self.active_positions) == 1
    assert fake_self.active_positions[0]["hold_days_remaining"] == 0

    # day 2: still quiet, hold_days_remaining<=0 -> force-closed as hold_expired
    KairosOrchestrator._manage_positions(fake_self, pd.Timestamp("2026-01-03"), histories)
    assert fake_self.active_positions == []
    assert len(fake_self.all_trades) == 1
    assert fake_self.all_trades[0].exit_reason == "hold_expired"


if __name__ == "__main__":
    test_unset_hold_days_never_force_closes_on_a_quiet_bar()
    test_explicit_hold_days_still_expires_as_before()
    print("ok")
