"""A non-finite bar must skip that symbol, not crash the whole group.

Regression test for the long-standing "cannot convert float NaN to integer"
failure: `KairosDistribution.from_bar` seeds its RNG with
`int(abs(close) * 1000)`, so a single NaN close anywhere in a symbol's history
raised ValueError out of `_make_realized_predictions` and aborted the entire
group's backtest. Because the bad bar is the "next bar" for exactly one date,
the failure was deterministic -- the same groups failed every sweep.
"""
import numpy as np
import pandas as pd
import pytest

from kairos_backtest import KairosSettings
from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig


def _frame(closes, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {"open": closes, "high": [c * 1.01 for c in closes],
         "low": [c * 0.99 for c in closes], "close": closes,
         "volume": [1000.0] * len(closes)},
        index=idx,
    )


def _orch():
    # no_prediction mode never calls predict_fn; it reads the realized next bar.
    return KairosOrchestrator(
        predict_fn=lambda *a, **kw: [],
        assets=["X"],
        config=OrchestratorConfig(no_prediction=True),
    )


@pytest.fixture(autouse=True)
def _small_samples(monkeypatch):
    monkeypatch.setattr(KairosSettings, "pred_samples", 8, raising=False)


def test_nan_close_in_next_bar_skips_symbol_not_crash():
    """The symbol whose next bar is NaN is dropped; the clean symbol survives."""
    good = _frame([10.0, 11.0, 12.0, 13.0])
    bad = _frame([20.0, 21.0, np.nan, 23.0])

    orch = _orch()
    orch._data_dict = {"GOOD": good, "BAD": bad}

    date = good.index[1]          # next bar is index 2 -> NaN for BAD
    histories = {"GOOD": good.loc[:date], "BAD": bad.loc[:date]}

    preds = orch._make_realized_predictions(date, histories)

    assert "GOOD" in preds
    assert "BAD" not in preds


def test_symbol_recovers_on_a_later_clean_date():
    """Skipping is per-date, not a permanent ban on the symbol."""
    bad = _frame([20.0, 21.0, np.nan, 23.0, 24.0])
    orch = _orch()
    orch._data_dict = {"BAD": bad}

    nan_date = bad.index[1]       # next bar NaN -> skipped
    assert orch._make_realized_predictions(nan_date, {"BAD": bad.loc[:nan_date]}) == {}

    ok_date = bad.index[3]        # next bar is 24.0 -> fine
    assert "BAD" in orch._make_realized_predictions(ok_date, {"BAD": bad.loc[:ok_date]})


def test_nonfinite_current_price_is_skipped():
    """A NaN entry price would silently poison every sampled close, not raise."""
    df = _frame([10.0, 11.0, np.nan, 13.0])
    orch = _orch()
    orch._data_dict = {"X": df}

    date = df.index[2]            # history ends on the NaN bar -> current_price NaN
    assert orch._make_realized_predictions(date, {"X": df.loc[:date]}) == {}


def test_infinite_bar_is_skipped():
    """inf overflows the same int() seed cast that NaN raises on."""
    df = _frame([10.0, 11.0, np.inf, 13.0])
    orch = _orch()
    orch._data_dict = {"X": df}

    date = df.index[1]
    assert orch._make_realized_predictions(date, {"X": df.loc[:date]}) == {}


def test_warning_is_emitted_once_per_symbol(capsys):
    df = _frame([10.0, np.nan, 12.0, np.nan, 14.0])
    orch = _orch()
    orch._data_dict = {"X": df}

    for i in (0, 2):
        d = df.index[i]
        orch._make_realized_predictions(d, {"X": df.loc[:d]})

    assert capsys.readouterr().out.count("non-finite OHLC bar") == 1


def test_clean_data_is_unaffected():
    """The guard must not change behaviour for well-formed input."""
    df = _frame([10.0, 11.0, 12.0, 13.0])
    orch = _orch()
    orch._data_dict = {"X": df}

    date = df.index[1]
    preds = orch._make_realized_predictions(date, {"X": df.loc[:date]})

    assert "X" in preds
    assert preds["X"].current_price == pytest.approx(11.0)
    # distribution is built from the *next* bar (12.0), anchored at entry
    assert np.isfinite(preds["X"].dist.stats["close"]["mean"])
