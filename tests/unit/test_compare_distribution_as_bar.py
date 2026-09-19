"""E20-S05: Unit tests for the offline distribution_as_bar vs. last_real_bar
comparison (scripts/compare_distribution_as_bar.py).

Crafted fixture proves the synthetic bar flips RSIFilterStrategy's decision
end-to-end (through KairosOrchestrator.run_backtest(), not a hand call to
generate_signal()), and that the comparison report correctly attributes the
resulting signal-count/EV delta to distribution_as_bar.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pandas as pd  # noqa: E402

from compare_distribution_as_bar import (  # noqa: E402
    SCOPED_STRATEGIES, _scoped_disabled_strategies, compare_prediction_usage_modes,
)


def _make_flip_fixture():
    """Crafted fixture where distribution_as_bar flips RSIFilterStrategy's
    LONG gate from "no signal" to "signal".

    History: a mild 3-up/10-down zigzag (RSI(14) == 33.33, just ABOVE the
    30.0 oversold threshold) -- last_real_bar mode reads this history
    unmodified, RSI stays >= 30, the oversold gate never opens, no signal.

    Oracle "future" bar: KairosDistribution.from_bar centers its truncated-
    normal sampling on current_price (the trade entry), not the peeked
    bar's own close -- see kairos_prediction_usage.py / from_bar's
    docstring. Setting the peeked bar's low bound only 1.0 below
    current_price (and a wide, distant high=150) makes the bulk of samples
    clip to that near boundary: pct_50 (what _build_synthetic_bar uses for
    the synthetic close) and pct_10 both land ~1 point below current_price,
    while the free upper tail keeps dist.stats["close"]["mean"] clearly
    above current_price (RSIFilterStrategy's direction gate) and gives a
    real, far pct_90 for a valid target -- a genuine stop-below/target-above
    bracket, not a degenerate one.

    Appending that ~-1 synthetic close as history's newest bar (only under
    distribution_as_bar) nudges RSI(14) down across the 30.0 line, opening
    the gate that last_real_bar never reaches. Verified empirically by
    running this exact fixture (not hand-derived from RSI arithmetic alone --
    from_bar's clip-then-percentile interaction is easier to confirm by
    running than to fully re-derive by hand).
    """
    deltas = [0.5, 0.5, 0.5] + [-0.3] * 10
    closes = [100.0]
    for d in deltas:
        closes.append(closes[-1] + d)

    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    history = pd.DataFrame({
        "open": closes, "high": [c * 1.001 for c in closes], "low": [c * 0.999 for c in closes],
        "close": closes, "volume": [1000.0] * len(closes),
    }, index=idx)
    current_price = closes[-1]

    # Oracle's peeked bar: date is irrelevant to sampling (only OHLC matters
    # to from_bar/_make_realized_predictions), placed one day after history.
    future_idx = idx[-1] + pd.Timedelta(days=1)
    future_bar = pd.DataFrame({
        "open": [105.0], "high": [150.0], "low": [current_price - 1.0],
        "close": [105.0], "volume": [1000.0],
    }, index=[future_idx])

    full_df = pd.concat([history, future_bar])
    # lookback = len(history) - 1 so common_dates covers exactly the crafted
    # date (history's last row, oracle peeks the appended future_bar) plus
    # the trailing date of future_bar itself (a degenerate self-referential
    # oracle evaluation with no directional edge -- verified empirically to
    # contribute no extra signals, see this story's implementation notes).
    return {"TEST": full_df}, len(history) - 1


def test_synthetic_bar_flips_rsi_filter_signal_count():
    """AC: crafted fixture where the synthetic bar flips an RSI-gated
    strategy's decision -- report shows a signal-count difference.
    """
    data_dict, lookback = _make_flip_fixture()
    report = compare_prediction_usage_modes(data_dict, ["TEST"], lookback=lookback)

    rsi_report = report["strategies"]["rsi_filter"]
    assert rsi_report["last_real_bar"]["signal_count"] == 0
    assert rsi_report["distribution_as_bar"]["signal_count"] == 1
    assert rsi_report["delta"]["signal_count"] == 1

    # EV delta is attributable to the same flip: last_real_bar has zero
    # resolved signals (ev_pct == 0.0, the empty-list default), while
    # distribution_as_bar's one signal resolves to a nonzero EV.
    assert rsi_report["last_real_bar"]["ev_pct"] == 0.0
    assert rsi_report["distribution_as_bar"]["ev_pct"] != 0.0
    assert rsi_report["delta"]["ev_pct"] != 0.0


def test_report_states_signal_generation_not_exit_resolution():
    """AC: report must state explicitly that this compares signal-generation
    behavior, distinguishing it from E18-S05's resolve-only comparison shape.
    """
    data_dict, lookback = _make_flip_fixture()
    report = compare_prediction_usage_modes(data_dict, ["TEST"], lookback=lookback)

    note = report["comparison_note"].lower()
    assert "signal-generation" in note
    assert "e18-s05" in note
    assert "run_backtest" in note


def test_scoped_disabled_strategies_excludes_only_the_two_scoped():
    """The disabled-strategies gate used to isolate the comparison must
    exclude exactly rsi_filter/macd_filter, disabling everything else."""
    disabled = _scoped_disabled_strategies()
    assert "rsi_filter" not in disabled
    assert "macd_filter" not in disabled
    # Sanity: a strategy known to be registered elsewhere IS disabled here.
    assert "skew" in disabled
    assert len(disabled) > 50


def test_scoped_strategies_constant():
    assert set(SCOPED_STRATEGIES) == {"rsi_filter", "macd_filter"}


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
