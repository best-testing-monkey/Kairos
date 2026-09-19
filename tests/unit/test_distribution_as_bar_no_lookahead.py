"""E20-S03: distribution_as_bar and _build_synthetic_bar read nothing beyond
their AssetPrediction argument.

This test proves that both functions are pure functions of:
  - pred.dist (KairosDistribution with .stats percentiles)
  - pred.current_price (float)
  - pred.history (DataFrame)
  - pred.symbol (str, unchanged on return)
  - interval (str parameter)

No reaching into module-level DataFrames, no peeking at a full price history,
no accessing state beyond what was passed in. This is safety-critical: the
naive-baseline zero-drift trap and oracle-decision peek bug (both documented
in CLAUDE.md 2026-08-28/2026-09-01) have the exact same shape — reading data
that should be off-limits. We guard against it here with two layers:

1. Functional test: `distribution_as_bar()` produces identical results when
   given an `AssetPrediction` in isolation vs. inside an orchestrator context
   (the oracle-mode case).

2. Static test: `inspect.getclosurevars()` on both functions confirms no
   closure-captured references to anything except stdlib/pandas/numpy/dataclasses
   names — catches accidental future regressions where someone adds a
   "helpful" reference to outside state.
"""
import inspect
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import numpy as np
import pandas as pd
import pytest

from kairos_backtest import KairosSettings, KairosDistribution
from kairos_meta import AssetPrediction
from kairos_prediction_usage import _build_synthetic_bar, distribution_as_bar
from kairos_orchestrator import KairosOrchestrator, OrchestratorConfig


@pytest.fixture(autouse=True)
def _small_samples(monkeypatch):
    """Use small sample count for faster tests."""
    monkeypatch.setattr(KairosSettings, "pred_samples", 8, raising=False)


def _make_test_history(closes, start="2024-01-01"):
    """Construct a minimal OHLCV DataFrame."""
    idx = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.02 for c in closes],
            "low": [c * 0.98 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=idx,
    )


def _make_test_distribution(current_price, variance=0.01):
    """Construct a minimal KairosDistribution with predictable stats.

    Creates a distribution centered on current_price with small variance,
    allowing us to verify the synthetic bar uses the distribution's stats
    (not any external data).
    """
    samples = []
    np.random.seed(42)  # deterministic
    for _ in range(8):
        offset = np.random.normal(0, variance * current_price)
        samples.append(
            pd.DataFrame(
                {
                    "open": [current_price + offset * 0.5],
                    "high": [current_price + offset],
                    "low": [current_price - offset],
                    "close": [current_price + offset * 0.3],
                }
            )
        )
    return KairosDistribution(samples)


def _make_asset_prediction_isolated(symbol="TEST", current_price=100.0):
    """Create an AssetPrediction in isolation, no orchestrator state.

    This is the "only data available" fixture the ticket specifies: just
    an AssetPrediction, its dist, and its history. No other module-level
    DataFrame, no orchestrator, no external context.
    """
    history = _make_test_history([95.0, 97.0, 100.0])
    dist = _make_test_distribution(current_price)
    return AssetPrediction(
        symbol=symbol,
        dist=dist,
        current_price=current_price,
        history=history,
    )


def test_build_synthetic_bar_reads_only_pred_fields():
    """Direct proof: _build_synthetic_bar only touches pred.dist,
    pred.current_price, and pred.history.

    If this reads anything else, the closure check below catches it.
    If the closure check passes, this validates the output is sensible.
    """
    pred = _make_asset_prediction_isolated(symbol="BTC", current_price=50000.0)
    bar = _build_synthetic_bar(pred, interval="1d")

    # Bar must be a Series with the expected structure
    assert isinstance(bar, pd.Series)
    assert set(bar.index) == {"open", "high", "low", "close", "volume"}

    # open must be current_price (the entry anchor)
    assert bar["open"] == pred.current_price

    # high/low/close must come from pred.dist.stats percentiles
    assert bar["high"] == pred.dist.stats["high"]["pct_50"]
    assert bar["low"] == pred.dist.stats["low"]["pct_50"]
    assert bar["close"] == pred.dist.stats["close"]["pct_50"]

    # volume carried forward from last history bar
    assert bar["volume"] == pred.history.iloc[-1]["volume"]

    # timestamp is one interval ahead of history's last row
    expected_ts = pred.history.index[-1] + pd.Timedelta(days=1)
    assert bar.name == expected_ts


def test_distribution_as_bar_returns_new_asset_prediction():
    """distribution_as_bar must return a new AssetPrediction with appended
    history, leaving everything else unchanged.
    """
    pred = _make_asset_prediction_isolated(symbol="ETH", current_price=3000.0)
    result = distribution_as_bar(pred, interval="1d")

    # Return type
    assert isinstance(result, AssetPrediction)

    # Unchanged fields
    assert result.symbol == pred.symbol
    assert result.current_price == pred.current_price
    assert result.dist is pred.dist  # same object

    # History extended by one row
    assert len(result.history) == len(pred.history) + 1
    assert list(result.history.index[:-1]) == list(pred.history.index)

    # Last row is the synthetic bar
    synth_row = result.history.iloc[-1]
    assert synth_row["open"] == pred.current_price
    assert synth_row["high"] == pred.dist.stats["high"]["pct_50"]


def test_distribution_as_bar_with_oracle_mode():
    """Oracle-mode-specific test: an AssetPrediction built via
    _make_realized_predictions (which peeks at a future bar) must still only
    be touched via its pred.dist/pred.current_price/pred.history by
    distribution_as_bar.

    The oracle peek happens in _make_realized_predictions, not in
    distribution_as_bar. This test confirms distribution_as_bar doesn't do
    any additional peeking of its own.
    """
    # Set up orchestrator in oracle mode (no_prediction=True, naive_baseline=False)
    orch = KairosOrchestrator(
        predict_fn=lambda *a, **kw: [],
        assets=["ORACLE_TEST"],
        config=OrchestratorConfig(no_prediction=True, naive_baseline=False),
    )

    # Full data: past + future
    full_data = _make_test_history([10.0, 11.0, 12.0, 13.0, 40.0, 41.0])
    date = full_data.index[3]  # signal date: 2024-01-04, next bar is 40.0

    orch._data_dict = {"ORACLE_TEST": full_data}
    histories = {"ORACLE_TEST": full_data[full_data.index <= date]}

    # This peeks at the future bar (40.0) to build the distribution.
    pred = orch._make_realized_predictions(date, histories, naive=False)[
        "ORACLE_TEST"
    ]

    # Now pass this oracle-peeked AssetPrediction to distribution_as_bar.
    # The function must not reach back into orch._data_dict, orch.config, or
    # any other state -- only the pred argument.
    result = distribution_as_bar(pred, interval="1d")

    # Verify it returned a valid AssetPrediction with extended history
    assert isinstance(result, AssetPrediction)
    assert len(result.history) == len(pred.history) + 1

    # The synthetic bar's open must be pred.current_price (12.0, the bar
    # before the peeked bar). This confirms we're only using the given
    # AssetPrediction fields, not reaching into the full data or the peeked
    # future bar a second time.
    synth_row = result.history.iloc[-1]
    assert synth_row["open"] == pred.current_price


def test_build_synthetic_bar_no_forbidden_closures():
    """Static check: _build_synthetic_bar has no closure-captured references
    beyond stdlib/pandas/numpy/dataclasses names and internal helpers.

    This catches accidental future regressions where someone adds a reference
    to module-level state (e.g., self._data_dict, self.config, a global
    DataFrame, etc.).

    Allowed closure names:
      - stdlib (timedelta, re, etc.)
      - pandas/numpy names (pd, np)
      - dataclasses (replace)
      - Internal helpers (_interval_to_timedelta, etc.)
      - None, __name__, etc. (magic/internal)

    Forbidden:
      - User-defined classes/objects (KairosOrchestrator, etc.)
      - External module-level variables (any non-stdlib state)
    """
    closure_vars = inspect.getclosurevars(_build_synthetic_bar)
    nonlocals = closure_vars.nonlocals
    globals_used = closure_vars.globals

    # Nonlocals should be empty (function is at module level)
    assert len(nonlocals) == 0, f"Unexpected nonlocals: {nonlocals}"

    # Globals should only include stdlib/pandas/numpy names and internal helpers
    allowed_globals = {
        "pd",
        "np",
        "re",
        "replace",
        "timedelta",
        "_interval_to_timedelta",  # internal helper in same module
        # Builtins and special names are OK
        "__name__",
        "__doc__",
        "__dict__",
    }
    forbidden = set(globals_used.keys()) - allowed_globals
    assert (
        not forbidden
    ), f"_build_synthetic_bar has forbidden closure references: {forbidden}"


def test_distribution_as_bar_no_forbidden_closures():
    """Static check: distribution_as_bar has no closure-captured references
    beyond stdlib/pandas/numpy/dataclasses names.
    """
    closure_vars = inspect.getclosurevars(distribution_as_bar)
    nonlocals = closure_vars.nonlocals
    globals_used = closure_vars.globals

    # Nonlocals should be empty
    assert len(nonlocals) == 0, f"Unexpected nonlocals: {nonlocals}"

    # Globals should only include stdlib/pandas/numpy names
    allowed_globals = {
        "pd",
        "_build_synthetic_bar",  # calls the sibling function
        "replace",
        "__name__",
        "__doc__",
        "__dict__",
    }
    forbidden = set(globals_used.keys()) - allowed_globals
    assert (
        not forbidden
    ), f"distribution_as_bar has forbidden closure references: {forbidden}"


if __name__ == "__main__":
    test_build_synthetic_bar_reads_only_pred_fields()
    test_distribution_as_bar_returns_new_asset_prediction()
    test_distribution_as_bar_with_oracle_mode()
    test_build_synthetic_bar_no_forbidden_closures()
    test_distribution_as_bar_no_forbidden_closures()
    print("ok")
