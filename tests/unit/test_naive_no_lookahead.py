"""The naive baseline must not see the future. Oracle must.

This is the invariant the naive stage exists for. Before 2026-09-01 it did not
hold: `_make_realized_predictions` built the distribution from the NEXT bar in
both modes, so naive shared oracle's perfect-foresight decision and differed
only in how the resulting trade was settled. `naive_baseline` reached exactly
one behavioural site (`_compute_shadow_performance`), never the decision.

The test is a direct probe: mutate only bars strictly after `date` and assert
the naive decision is byte-identical. The oracle case is the control -- it must
change, or the probe has no power and would pass on a broken implementation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import numpy as np
import pandas as pd
import pytest

from kairos_backtest import KairosSettings
from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig


@pytest.fixture(autouse=True)
def _small_samples(monkeypatch):
    monkeypatch.setattr(KairosSettings, "pred_samples", 8, raising=False)


def _frame(closes, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {"open": closes, "high": [c * 1.02 for c in closes],
         "low": [c * 0.98 for c in closes], "close": closes,
         "volume": [1000.0] * len(closes)},
        index=idx,
    )


def _orch(naive):
    return KairosOrchestrator(
        predict_fn=lambda *a, **kw: [],
        assets=["X"],
        config=OrchestratorConfig(no_prediction=True, naive_baseline=naive),
    )


def _predict(naive, closes, date_pos):
    """Run one prediction step and return everything the strategy would see."""
    df = _frame(closes)
    date = df.index[date_pos]
    orch = _orch(naive)
    orch._data_dict = {"X": df}
    histories = {"X": df[df.index <= date]}
    pred = orch._make_realized_predictions(date, histories, naive=naive)["X"]
    closes_sampled = tuple(round(float(r["close"].iloc[0]), 10) for r in pred.dist.predictions)
    return {
        "current_price": pred.current_price,
        "history_last": pred.history.index[-1],
        "samples": closes_sampled,
    }


# Same past, wildly different future. Only bars after index 3 differ.
PAST = [10.0, 11.0, 12.0, 13.0]
FUTURE_UP = PAST + [40.0, 41.0]
FUTURE_DOWN = PAST + [2.0, 1.0]


def test_naive_decision_ignores_every_bar_after_the_signal_date():
    up = _predict(naive=True, closes=FUTURE_UP, date_pos=3)
    down = _predict(naive=True, closes=FUTURE_DOWN, date_pos=3)
    assert up == down, "naive peeked: the future changed its decision inputs"


def test_oracle_decision_does_depend_on_the_next_bar():
    """Control. If this ever passes-as-equal the probe above is worthless."""
    up = _predict(naive=False, closes=FUTURE_UP, date_pos=3)
    down = _predict(naive=False, closes=FUTURE_DOWN, date_pos=3)
    assert up["samples"] != down["samples"]


def test_naive_withholds_the_last_bar_from_the_strategy():
    """History stops one bar short, and the entry price is that earlier close."""
    naive = _predict(naive=True, closes=FUTURE_UP, date_pos=3)
    oracle = _predict(naive=False, closes=FUTURE_UP, date_pos=3)

    idx = _frame(FUTURE_UP).index
    assert naive["history_last"] == idx[2]   # bar 3 withheld
    assert oracle["history_last"] == idx[3]  # oracle sees it
    assert naive["current_price"] == 12.0    # close of bar 2
    assert oracle["current_price"] == 13.0   # close of bar 3


def test_naive_forecast_is_the_withheld_bar_so_it_carries_a_real_move():
    """The withheld bar's range, not a distribution centred on the entry.

    Centring on the entry price is the removed `use_current_bar` mode, which
    baked in zero drift and handicapped every directional strategy. Here bar 3
    (close 13.0, low 12.74) sits entirely above the entry of 12.0, so every
    sampled close must too -- a genuine, already-realised upward move.
    """
    naive = _predict(naive=True, closes=FUTURE_UP, date_pos=3)
    assert all(s > naive["current_price"] for s in naive["samples"])


def test_naive_needs_two_bars_and_skips_the_symbol_rather_than_crashing():
    df = _frame([10.0, 11.0])
    date = df.index[0]
    orch = _orch(naive=True)
    orch._data_dict = {"X": df}
    assert orch._make_realized_predictions(date, {"X": df[df.index <= date]}, naive=True) == {}


if __name__ == "__main__":
    test_naive_decision_ignores_every_bar_after_the_signal_date()
    test_oracle_decision_does_depend_on_the_next_bar()
    test_naive_withholds_the_last_bar_from_the_strategy()
    test_naive_forecast_is_the_withheld_bar_so_it_carries_a_real_move()
    test_naive_needs_two_bars_and_skips_the_symbol_rather_than_crashing()
    print("ok")


def test_context_returns_window_also_withholds_the_bar():
    """context["returns_window"] is part of "what the strategy gets".

    If it kept the withheld bar, a strategy reading it would see the forecast
    directly instead of through the distribution, making the forecast trivially
    self-fulfilling. Oracle is unaffected: its forecast bar is never in history
    to begin with.
    """
    import types

    df = _frame([10.0, 11.0, 12.0, 13.0, 40.0])
    date = df.index[3]

    seen = {}
    for naive in (True, False):
        orch = _orch(naive)
        orch._data_dict = {"X": df}
        orch.strategies = []  # no strategies: we only want the context inputs
        captured = {}
        real = orch._compute_returns_window

        def spy(histories, _real=real, _c=captured):
            _c["last_index"] = {s: h.index[-1] for s, h in histories.items()}
            return _real(histories)

        orch._compute_returns_window = spy
        orch._run_day(date, {"X": df[df.index <= date]})
        seen[naive] = captured["last_index"]["X"]

    assert seen[True] == df.index[2]    # naive: withheld bar excluded
    assert seen[False] == df.index[3]   # oracle: full history
