"""Oracle/base shadow evaluation walks forward multi-bar, same as naive.

Before 2026-08-29 `_compute_shadow_performance` checked exactly one bar and
force-closed at that bar's close, while the naive path walked forward until a
genuine trigger. Floor and ceiling were measured with different rulers. These
tests pin the shared rule: hold through non-triggering bars, exclude a signal
that never resolves, and treat entry anchoring as the ONLY difference between
the two modes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from types import SimpleNamespace
import pandas as pd

from kairos_orchestrator import KairosOrchestrator
from kairos_backtest import Direction


def _bar(o, h, l, c):
    return {"open": o, "high": h, "low": l, "close": c}


def _df(date0, bars):
    return pd.DataFrame(bars, index=[date0 + pd.Timedelta(days=i) for i in range(len(bars))])


def _run(signal, bars, naive=False):
    date0 = signal[0]
    fake = SimpleNamespace(_shadow_signals=[signal], _data_dict={"TEST": _df(date0, bars)})
    return KairosOrchestrator._compute_shadow_performance(fake, naive=naive)


D0 = pd.Timestamp("2026-01-01")
# LONG, decided at D0: stop -5%, target +10% relative to a reference entry of 100
LONG_SIG = (D0, "TEST", "s", Direction.LONG, 95.0, 110.0, 100.0)


def test_oracle_holds_past_a_nontriggering_bar():
    """The regression this change is about: bar 1 triggers nothing, so hold."""
    res = _run(LONG_SIG, [
        _bar(99, 101, 98, 100),      # D0, the decision bar itself
        _bar(100, 104, 98, 102),     # entry at open=100 -> stop 95, target 110; neither hit
        _bar(103, 106, 101, 105),    # still nothing
        _bar(105, 112, 104, 111),    # high 112 >= 110 -> target
    ])
    pnl = res["s"]["pnl_list"][0]
    assert abs(pnl - 0.10) < 1e-9        # exits at the target, 100 -> 110
    assert abs(pnl - 0.02) > 1e-9        # NOT the old force-close at bar 1's close of 102


def test_oracle_unresolved_is_excluded_not_force_closed():
    res = _run(LONG_SIG, [
        _bar(99, 101, 98, 100),
        _bar(100, 104, 98, 102),
        _bar(102, 105, 99, 103),     # data ends with neither stop nor target hit
    ])
    assert "s" not in res            # excluded outright, no invented close-out


def test_oracle_gap_open_exits_at_the_open_on_a_later_bar():
    """Gap handling is live from the bar AFTER entry onward."""
    res = _run(LONG_SIG, [
        _bar(99, 101, 98, 100),
        _bar(100, 104, 98, 102),     # entry at 100
        _bar(120, 121, 119, 120),    # opens straight through the 110 target
    ])
    pnl = res["s"]["pnl_list"][0]
    assert abs(pnl - 0.20) < 1e-9    # filled at the gap open of 120, not the nominal 110


def test_short_signals_walk_forward_too():
    short_sig = (D0, "TEST", "s", Direction.SHORT, 105.0, 90.0, 100.0)
    res = _run(short_sig, [
        _bar(99, 101, 98, 100),
        _bar(100, 103, 97, 99),      # entry at open=100 -> stop 105, target 90; neither hit
        _bar(98, 101, 88, 89),       # low 88 <= 90 -> target
    ])
    pnl = res["s"]["pnl_list"][0]
    assert abs(pnl - 0.10) < 1e-9    # short 100 -> 90
    assert abs(pnl - 0.01) > 1e-9    # NOT the old force-close at bar 1's close of 99


def test_both_modes_now_share_one_exit_rule():
    """Same signal, same bars: only the entry anchor differs, not the exit rule.

    Oracle fills at bar 1's open (100); naive cannot exist until bar 1 has
    closed, so it fills at 102. Both then walk forward to the same +10%
    target and book the same return.
    """
    bars = [
        _bar(99, 101, 98, 100),
        _bar(100, 104, 98, 102),     # oracle entry 100 | naive entry 102
        _bar(103, 115, 102, 113),    # target: 110 for oracle, 112.2 for naive -- both hit
    ]
    oracle = _run(LONG_SIG, bars)["s"]["pnl_list"][0]
    naive = _run(LONG_SIG, bars, naive=True)["s"]["pnl_list"][0]

    assert abs(oracle - 0.10) < 1e-9
    assert abs(naive - 0.10) < 1e-9


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_"):
            _fn()
    print("ok")
