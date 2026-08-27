"""_make_realized_predictions: oracle (future peek) vs. naive (current bar) branch."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from types import SimpleNamespace
import pandas as pd

from kairos_orchestrator import KairosOrchestrator


def _bar(o, h, l, c, v=1000.0):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


def _make_fake_self(use_current_bar):
    history = pd.DataFrame(
        [_bar(100, 105, 95, 100)],
        index=[pd.Timestamp("2026-01-01")],
    )
    future = pd.DataFrame(
        [_bar(100, 500, 50, 100)],  # deliberately wide range, easy to distinguish
        index=[pd.Timestamp("2026-01-02")],
    )
    full_df = pd.concat([history, future])
    return SimpleNamespace(
        config=SimpleNamespace(use_current_bar=use_current_bar),
        _data_dict={"TEST": full_df},
    ), history


def test_oracle_mode_uses_future_bar():
    fake_self, history = _make_fake_self(use_current_bar=False)
    result = KairosOrchestrator._make_realized_predictions(
        fake_self, pd.Timestamp("2026-01-01"), {"TEST": history}
    )
    dist = result["TEST"].dist
    assert dist.df["high"].max() > 200  # only reachable from the wide future bar


def test_naive_mode_uses_current_bar_only():
    fake_self, history = _make_fake_self(use_current_bar=True)
    result = KairosOrchestrator._make_realized_predictions(
        fake_self, pd.Timestamp("2026-01-01"), {"TEST": history}
    )
    dist = result["TEST"].dist
    assert dist.df["high"].max() <= 105.0001  # bounded by the current bar's own range
    assert result["TEST"].current_price == 100.0


if __name__ == "__main__":
    test_oracle_mode_uses_future_bar()
    test_naive_mode_uses_current_bar_only()
    print("ok")
