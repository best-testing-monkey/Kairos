"""_compute_shadow_performance_naive: anchor entry to the WITHHELD bar's close,
walk forward until a genuine stop/target trigger, exclude signals that never
resolve.

The withheld bar is the last bar at or before the signal's `date` -- the bar
`_make_realized_predictions` handed the strategy as its forecast. It has already
closed by the time the trade can be placed, so entering at its close is
actionable, and every bar after it is genuinely unseen.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from types import SimpleNamespace
import pandas as pd

from kairos_orchestrator import KairosOrchestrator
from kairos_backtest import Direction


def _bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def _fake_self(shadow_signals, data_dict):
    return SimpleNamespace(_shadow_signals=shadow_signals, _data_dict=data_dict)


def test_target_hit_after_holding_two_bars_uses_reanchored_entry():
    # the strategy's own decision: LONG, sig_entry=100, stop=95, target=110
    date0 = pd.Timestamp("2026-01-01")
    signal = (date0, "TEST", "test_strat", Direction.LONG, 95.0, 110.0, 100.0)

    df = pd.DataFrame(
        [
            _bar(99, 101, 98, 100),    # date0: the withheld bar -> entry = its close, 100
            _bar(102, 106, 101, 104),  # held: nothing triggers (101 > 95, 106 < 110)
            _bar(105, 107, 103, 106),  # held: nothing triggers
            _bar(106, 115, 104, 112),  # target hit intrabar (115 >= 110)
        ],
        index=[date0 + pd.Timedelta(days=i) for i in range(4)],
    )

    fake_self = _fake_self([signal], {"TEST": df})
    result = KairosOrchestrator._compute_shadow_performance_naive(fake_self)

    assert result["test_strat"]["signal_count"] == 1
    pnl = result["test_strat"]["pnl_list"][0]
    # entry=100, exit=target_price=110 -> pnl=(110-100)/100=0.1
    assert abs(pnl - 0.1) < 1e-9


def test_entry_is_the_withheld_bar_close_not_the_bar_after_it():
    """Pins the anchoring itself, which the proportional case above cannot.

    Stop/target are re-anchored proportionally, so a test whose bars move by a
    round percentage gives the same pnl under either anchoring. Here the exit is
    an absolute gap price, so the entry actually shows up in the answer.
    """
    date0 = pd.Timestamp("2026-05-01")
    signal = (date0, "TEST", "test_strat", Direction.LONG, 95.0, 110.0, 100.0)

    df = pd.DataFrame(
        [
            _bar(99, 101, 98, 100),     # withheld bar -> entry=100
            _bar(150, 151, 149, 150),   # gaps through target on the very next bar
        ],
        index=[date0, date0 + pd.Timedelta(days=1)],
    )

    fake_self = _fake_self([signal], {"TEST": df})
    pnl = KairosOrchestrator._compute_shadow_performance_naive(fake_self)["test_strat"]["pnl_list"][0]

    # entry=100 (withheld bar's close) -> 0.5. Anchoring to the NEXT bar's close
    # (150) would make this trade impossible to enter and exit at 150 for 0.0.
    assert abs(pnl - 0.5) < 1e-9


def test_never_resolves_is_excluded_not_force_closed():
    date0 = pd.Timestamp("2026-02-01")
    signal = (date0, "TEST", "test_strat", Direction.LONG, 95.0, 110.0, 100.0)

    df = pd.DataFrame(
        [
            _bar(99, 101, 98, 100),    # withheld bar -> entry=100, stop=95, target=110
            _bar(102, 106, 101, 104),  # never triggers
            _bar(103, 105, 102, 104),  # never triggers
            _bar(104, 106, 103, 105),  # never triggers -- data ends here
        ],
        index=[date0 + pd.Timedelta(days=i) for i in range(4)],
    )

    fake_self = _fake_self([signal], {"TEST": df})
    result = KairosOrchestrator._compute_shadow_performance_naive(fake_self)

    assert "test_strat" not in result  # excluded entirely, no fake close-out


def test_open_gap_triggers_immediately():
    date0 = pd.Timestamp("2026-03-01")
    signal = (date0, "TEST", "test_strat", Direction.LONG, 95.0, 110.0, 100.0)

    df = pd.DataFrame(
        [
            _bar(99, 101, 98, 100),     # withheld bar -> entry=100, stop=95, target=110
            _bar(102, 106, 101, 104),   # no trigger (101 > 95, 106 < 110)
            _bar(120, 121, 119, 120),   # opens straight through target
        ],
        index=[date0 + pd.Timedelta(days=i) for i in range(3)],
    )

    fake_self = _fake_self([signal], {"TEST": df})
    result = KairosOrchestrator._compute_shadow_performance_naive(fake_self)

    pnl = result["test_strat"]["pnl_list"][0]
    # gap-open exit at 120, not the nominal target price of 110
    assert abs(pnl - (120.0 - 100.0) / 100.0) < 1e-9


if __name__ == "__main__":
    test_target_hit_after_holding_two_bars_uses_reanchored_entry()
    test_entry_is_the_withheld_bar_close_not_the_bar_after_it()
    test_never_resolves_is_excluded_not_force_closed()
    test_open_gap_triggers_immediately()
    print("ok")
