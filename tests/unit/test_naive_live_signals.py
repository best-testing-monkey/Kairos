"""The naive baseline, generated live instead of swept.

`--naive` makes papertrade trade the persistence forecast: withhold the last
completed bar from the history the strategy sees and hand it back as the
forecast. These tests pin the three properties that make that honest and
tradeable, each of which has already been a real bug once:

  * the withholding actually happens, and reaches the CONTEXT too, not just
    AssetPrediction.history (CLAUDE.md, "Naive baseline" -- leaving
    returns_window untruncated let a strategy read the forecast bar directly
    and made the forecast self-fulfilling);
  * re-anchoring moves the signal onto the withheld bar's close without
    changing the decision, so the bracket still brackets the fill (CLAUDE.md,
    "Stale-signal-cache brackets", commit 7cc66d4, is what rejects it
    otherwise);
  * a naive pass and a base pass do not share signals_cache entries.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import datetime as _dt

import numpy as np
import pandas as pd
import pytest

from kairos_backtest import KairosSettings, Signal, Direction
import kairos_signals as ks


@pytest.fixture(autouse=True)
def _small_samples(monkeypatch):
    monkeypatch.setattr(KairosSettings, "pred_samples", 16, raising=False)


def _frame(n=40, seed=0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 1, n))
    return pd.DataFrame({
        "open": close, "high": close + 2.0, "low": close - 2.0,
        "close": close, "volume": 1e6, "amount": 1e8,
    }, index=idx)


# --------------------------------------------------------------------------
# Withholding
# --------------------------------------------------------------------------

def test_last_bar_is_withheld_from_history():
    df = _frame()
    pred = ks._naive_predict_fn({"TEST": df})["TEST"]

    assert len(pred.history) == len(df) - 1
    assert pred.history.index[-1] == df.index[-2]
    # The price the decision is made against is the bar BEFORE the withheld
    # one. Centring on the withheld bar's own close instead is the removed
    # `use_current_bar` zero-drift trap.
    assert pred.current_price == pytest.approx(float(df["close"].iloc[-2]))
    assert pred.current_price != pytest.approx(float(df["close"].iloc[-1]))


def test_withholding_reaches_the_context_not_just_the_history():
    """The bug that survived the first naive rebuild for a few hours."""
    from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig

    df = _frame()
    assets = {"TEST": df}
    preds = ks._naive_predict_fn(assets)
    orch = KairosOrchestrator(
        predict_fn=lambda *a, **kw: [], assets=["TEST"],
        config=OrchestratorConfig.for_interval("1d"),
    )
    ctx = ks._build_context(orch, "TEST", preds["TEST"].current_price,
                            preds, preds["TEST"].history)

    # returns_window is derived from pred.history, so it must stop one bar
    # short of the frame -- a strategy reading it cannot see the forecast bar.
    assert ctx["returns_window"].index[-1] == df.index[-2]
    assert df.index[-1] not in ctx["returns_window"].index
    assert ctx["date"] == df.index[-2]


def test_naive_prediction_uses_only_the_frame_it_is_given():
    """Live-computability: no _data_dict, no lookahead, nothing off `self`.

    Truncating the frame by one bar must give exactly what the untruncated
    frame gave one bar earlier -- the definition of a forecast that could
    have been made at the time.
    """
    df = _frame()
    a = ks._naive_predict_fn({"TEST": df.iloc[:-1]})["TEST"]
    b = ks._naive_predict_fn({"TEST": df})["TEST"]

    assert a.current_price == pytest.approx(float(df["close"].iloc[-3]))
    assert b.current_price == pytest.approx(float(df["close"].iloc[-2]))
    assert len(a.history) == len(b.history) - 1


def test_symbol_with_one_bar_is_skipped_not_crashed():
    df = _frame(n=1)
    assert ks._naive_predict_fn({"TEST": df}) == {}


# --------------------------------------------------------------------------
# Re-anchoring
# --------------------------------------------------------------------------

def _sig(entry=100.0, stop=95.0, target=110.0, ev=5.0, direction=Direction.LONG):
    return Signal(direction=direction, size=1.0, entry=entry, stop=stop,
                  target=target, strategy_name="t", confidence=0.6,
                  expected_value=ev)


@pytest.mark.parametrize("fill", [102.0, 98.0, 100.0])
def test_reanchor_preserves_the_decision(fill):
    sig = _sig()
    before = (sig.stop / sig.entry, sig.target / sig.entry,
              sig.expected_value / sig.entry)

    ks._reanchor_naive_signal(sig, fill)

    assert sig.entry == fill
    after = (sig.stop / sig.entry, sig.target / sig.entry,
             sig.expected_value / sig.entry)
    # Only the accounting moves. EV-as-%-of-entry in particular must be
    # invariant, or min_ev_pct would gate a different edge than the strategy
    # actually expressed.
    assert after == pytest.approx(before)


def test_reanchor_keeps_the_bracket_around_the_fill():
    """The whole point: an un-anchored bracket is rejected at fill time."""
    sig = _sig()
    ks._reanchor_naive_signal(sig, 108.0)
    assert sig.stop < sig.entry < sig.target


def test_reanchor_matches_the_shadow_evaluators_formula():
    """Same numbers as KairosOrchestrator._compute_shadow_performance(naive=True)."""
    sig = _sig(entry=100.0, stop=95.0, target=110.0)
    ref, fill = sig.entry, 103.0
    expected_stop = fill * (1.0 + (sig.stop - ref) / ref)
    expected_target = fill * (1.0 + (sig.target - ref) / ref)

    ks._reanchor_naive_signal(sig, fill)

    assert sig.stop == pytest.approx(expected_stop)
    assert sig.target == pytest.approx(expected_target)


@pytest.mark.parametrize("entry,fill", [(0.0, 100.0), (100.0, 0.0), (100.0, -1.0)])
def test_reanchor_is_a_noop_on_unusable_prices(entry, fill):
    sig = _sig(entry=entry)
    stop, target = sig.stop, sig.target
    ks._reanchor_naive_signal(sig, fill)
    assert (sig.entry, sig.stop, sig.target) == (entry, stop, target)


# --------------------------------------------------------------------------
# Cache separation
# --------------------------------------------------------------------------

def test_naive_and_base_do_not_share_a_signals_cache_key():
    args = ("strat", "A,B", "1d", _dt.date(2026, 9, 1), 300, 100, 0.1, None, "")
    assert ks._signals_cache_key(*args, naive=True) != ks._signals_cache_key(*args)


def test_existing_base_keys_are_unchanged():
    """naive occupies the model slot, so no already-cached key is invalidated."""
    args = ("strat", "A,B", "1d", _dt.date(2026, 9, 1), 300, 100, 0.1, None, "")
    assert ks._signals_cache_key(*args) == "|".join(
        ["strat", "A,B", "1d", "2026-09-01", "300", "100", "0.1", "base", ""]
    )
