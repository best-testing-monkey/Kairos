"""_compute_shadow_performance_naive: re-anchor entry, walk forward until a
genuine stop/target trigger, exclude signals that never resolve."""
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
    # oracle's original decision: LONG, sig_entry=100, stop=95, target=110
    date0 = pd.Timestamp("2026-01-01")
    signal = (date0, "TEST", "test_strat", Direction.LONG, 95.0, 110.0, 100.0)

    df = pd.DataFrame(
        [
            _bar(99, 101, 98, 100),   # date0 (decision day, not used directly)
            _bar(102, 106, 101, 104),  # entry bar oracle peeked at -> new entry=104
            _bar(105, 107, 103, 106),  # held: nothing triggers (98.8 < 103, 107 < 114.4)
            _bar(106, 115, 104, 112),  # target hit intrabar (115 >= 114.4)
        ],
        index=[date0, date0 + pd.Timedelta(days=1), date0 + pd.Timedelta(days=2), date0 + pd.Timedelta(days=3)],
    )

    fake_self = _fake_self([signal], {"TEST": df})
    result = KairosOrchestrator._compute_shadow_performance_naive(fake_self)

    assert result["test_strat"]["signal_count"] == 1
    pnl = result["test_strat"]["pnl_list"][0]
    # entry=104, exit=target_price=104*1.10=114.4 -> pnl=(114.4-104)/104=0.1
    assert abs(pnl - 0.1) < 1e-9


def test_never_resolves_is_excluded_not_force_closed():
    date0 = pd.Timestamp("2026-02-01")
    signal = (date0, "TEST", "test_strat", Direction.LONG, 95.0, 110.0, 100.0)

    df = pd.DataFrame(
        [
            _bar(99, 101, 98, 100),
            _bar(102, 106, 101, 104),  # entry bar -> new entry=104, stop=98.8, target=114.4
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
            _bar(99, 101, 98, 100),
            _bar(102, 106, 101, 104),   # entry bar -> new entry=104, stop=98.8, target=114.4
            _bar(120, 121, 119, 120),   # opens straight through target
        ],
        index=[date0 + pd.Timedelta(days=i) for i in range(3)],
    )

    fake_self = _fake_self([signal], {"TEST": df})
    result = KairosOrchestrator._compute_shadow_performance_naive(fake_self)

    pnl = result["test_strat"]["pnl_list"][0]
    # gap-open exit at 120, not the nominal target price of 114.4
    assert abs(pnl - (120.0 - 104.0) / 104.0) < 1e-9


if __name__ == "__main__":
    test_target_hit_after_holding_two_bars_uses_reanchored_entry()
    test_never_resolves_is_excluded_not_force_closed()
    test_open_gap_triggers_immediately()
    print("ok")
