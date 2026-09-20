"""E21-S01: two bugs at the PREDICTION_USAGE_MODES dispatch call site in
_run_day() (strategy/kairos_orchestrator.py), found by code review of the
E18/E19/E20 rollout.

Bug 1 (lookahead): OrchestratorConfig(naive_baseline=True,
prediction_usage_mode=<non-identity>) used to silently leak an approximation
of the withheld bar back into context_histories/returns_window/realized_vol
via distribution_as_bar's synthetic bar -- reopening the exact lookahead bug
this project already fixed twice (see CLAUDE.md's "Oracle vs. naive-baseline
modes"). Fixed by rejecting the combination at OrchestratorConfig construction
(dataclass __post_init__), since the composition isn't meaningful in the
first place: naive mode exists specifically to prove a strategy needs no
future information, and re-injecting a guess at the withheld bar defeats that
by construction.

Bug 2 (interval): _run_day()'s dispatch call `usage_fn(pred)` never passed an
interval, so distribution_as_bar's interval="1d" default silently applied on
every backtest regardless of the real configured interval -- corrupting bar
spacing (RSI/MACD/ATR) on any 1h/4h backtest. Fixed by threading
KairosSettings.interval (the same source of truth __init__/
_compute_shadow_performance/_build_results already use for "the real
interval" in this class) through the dispatch call.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from kairos.errors import ConfigError  # noqa: E402
from kairos_backtest import KairosSettings, KairosDistribution  # noqa: E402
from kairos_meta import AssetPrediction  # noqa: E402
from kairos_orchestrator import (  # noqa: E402
    KairosOrchestrator, OrchestratorConfig, PREDICTION_USAGE_MODES,
)


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


def _make_pred(current_price=100.0):
    """A minimal, self-contained AssetPrediction for direct dispatch calls."""
    history = _frame([95.0, 97.0, 100.0])
    bar = pd.Series({
        "open": current_price, "high": current_price * 1.02,
        "low": current_price * 0.98, "close": current_price * 1.01,
        "volume": 1000.0,
    })
    dist = KairosDistribution.from_bar(bar, n_samples=8, center=current_price)
    return AssetPrediction(symbol="X", dist=dist, current_price=current_price, history=history)


class _CaptureStrategy:
    """Strategy stand-in: records the history it's handed, fires no signal.

    Used to observe the post-dispatch AssetPrediction.history that reaches
    strategy code inside _run_day(), without depending on any real strategy's
    gating logic.
    """
    name = "capture"

    def __init__(self):
        self.seen_last_index = None
        self.calls = 0

    def generate_signal(self, dist, current_price, history, context):
        self.seen_last_index = history.index[-1]
        self.calls += 1
        return None


# ---------------------------------------------------------------------------
# Bug 1: naive_baseline + non-identity prediction_usage_mode is rejected
# ---------------------------------------------------------------------------

class TestNaiveBaselineRejectsNonIdentityMode:
    def test_naive_plus_distribution_as_bar_raises_config_error(self):
        with pytest.raises(ConfigError, match="naive_baseline"):
            OrchestratorConfig(naive_baseline=True, prediction_usage_mode="distribution_as_bar")

    def test_error_message_names_the_offending_mode(self):
        with pytest.raises(ConfigError, match="distribution_as_bar"):
            OrchestratorConfig(naive_baseline=True, prediction_usage_mode="distribution_as_bar")

    def test_for_interval_construction_path_also_enforces_the_guard(self):
        """OrchestratorConfig.for_interval() constructs via cls(**merged), so
        the same __post_init__ guard must fire through that path too."""
        with pytest.raises(ConfigError):
            OrchestratorConfig.for_interval(
                "1h", naive_baseline=True, prediction_usage_mode="distribution_as_bar",
            )

    def test_naive_plus_default_mode_is_still_allowed(self):
        """Regression: naive_baseline=True with the default (identity) mode
        is the normal, load-bearing naive-baseline use case and must not raise."""
        config = OrchestratorConfig(naive_baseline=True)
        assert config.prediction_usage_mode == "last_real_bar"

    def test_naive_plus_explicit_identity_mode_is_allowed(self):
        config = OrchestratorConfig(naive_baseline=True, prediction_usage_mode="last_real_bar")
        assert config.naive_baseline is True

    def test_non_naive_plus_distribution_as_bar_is_allowed(self):
        """distribution_as_bar is fine outside naive mode (oracle/model) --
        only the naive_baseline composition is rejected."""
        config = OrchestratorConfig(naive_baseline=False, prediction_usage_mode="distribution_as_bar")
        assert config.prediction_usage_mode == "distribution_as_bar"


# ---------------------------------------------------------------------------
# Bug 2: the dispatch call threads the real interval, not a hardcoded "1d"
# ---------------------------------------------------------------------------

class TestIntervalThreadedToDispatch:
    def test_distribution_as_bar_dispatch_uses_real_interval_not_1d(self, monkeypatch):
        """Dispatch distribution_as_bar through _run_day with interval="1h"
        and assert the synthetic bar lands exactly one hour after the last
        real bar, not one day (the old hardcoded default)."""
        monkeypatch.setattr(KairosSettings, "interval", "1h", raising=False)

        orch = KairosOrchestrator(
            predict_fn=lambda *a, **kw: [],
            assets=["X"],
            config=OrchestratorConfig(
                no_prediction=True, naive_baseline=False,
                prediction_usage_mode="distribution_as_bar",
                # Meta-filters are irrelevant to this bug and would otherwise
                # depend on entropy/kurtosis noise from the small sample count.
                entropy_threshold=999.0, bimodality_filter=False,
            ),
        )
        df = _frame([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        date = df.index[3]
        orch._data_dict = {"X": df}
        capture = _CaptureStrategy()
        orch.strategies = [capture]

        orch._run_day(date, {"X": df[df.index <= date]})

        assert capture.calls == 1
        assert capture.seen_last_index == date + pd.Timedelta(hours=1)
        assert capture.seen_last_index != date + pd.Timedelta(days=1)

    def test_distribution_as_bar_dispatch_still_uses_1d_by_default(self, monkeypatch):
        """Control: with the real interval left at its "1d" default, the
        synthetic bar should land one day ahead, same as before this fix."""
        monkeypatch.setattr(KairosSettings, "interval", "1d", raising=False)

        orch = KairosOrchestrator(
            predict_fn=lambda *a, **kw: [],
            assets=["X"],
            config=OrchestratorConfig(
                no_prediction=True, naive_baseline=False,
                prediction_usage_mode="distribution_as_bar",
                entropy_threshold=999.0, bimodality_filter=False,
            ),
        )
        df = _frame([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
        date = df.index[3]
        orch._data_dict = {"X": df}
        capture = _CaptureStrategy()
        orch.strategies = [capture]

        orch._run_day(date, {"X": df[df.index <= date]})

        assert capture.seen_last_index == date + pd.Timedelta(days=1)


# ---------------------------------------------------------------------------
# Regression: "last_real_bar" (identity mode) still works unchanged
# ---------------------------------------------------------------------------

class TestIdentityModeSignatureRegression:
    def test_identity_mode_single_arg_call_still_works(self):
        """Pre-E21-S01 callers invoked the registry entry with just `pred`;
        the signature change (adding `interval`) must not break that."""
        pred = _make_pred()
        identity_fn = PREDICTION_USAGE_MODES["last_real_bar"]
        result = identity_fn(pred)

        assert result is not pred
        assert result.symbol == pred.symbol
        assert result.current_price == pred.current_price
        assert result.dist is pred.dist
        assert result.history is pred.history

    def test_identity_mode_ignores_interval_arg(self):
        pred = _make_pred()
        identity_fn = PREDICTION_USAGE_MODES["last_real_bar"]
        result_1d = identity_fn(pred, "1d")
        result_1h = identity_fn(pred, "1h")

        assert result_1d.history is pred.history
        assert result_1h.history is pred.history
        assert result_1d.current_price == result_1h.current_price == pred.current_price

    def test_identity_mode_dispatch_through_run_day_unaffected_by_interval(self, monkeypatch):
        """End-to-end: default prediction_usage_mode="last_real_bar" through
        _run_day() must still leave history untouched regardless of the real
        interval -- the identity mode has nothing to advance a timestamp by."""
        monkeypatch.setattr(KairosSettings, "interval", "1h", raising=False)

        orch = KairosOrchestrator(
            predict_fn=lambda *a, **kw: [],
            assets=["X"],
            config=OrchestratorConfig(
                no_prediction=True, naive_baseline=False,
                entropy_threshold=999.0, bimodality_filter=False,
            ),
        )
        assert orch.config.prediction_usage_mode == "last_real_bar"
        df = _frame([10.0, 11.0, 12.0, 13.0])
        date = df.index[-1]
        orch._data_dict = {"X": df}
        capture = _CaptureStrategy()
        orch.strategies = [capture]

        orch._run_day(date, {"X": df[df.index <= date]})

        assert capture.calls == 1
        assert capture.seen_last_index == date  # unchanged, no synthetic bar appended


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
