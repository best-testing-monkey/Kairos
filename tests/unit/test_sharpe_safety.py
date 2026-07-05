"""Tests for the robust Sharpe-ratio computation in kairos_orchestrator.

Guards against the shadow-backtest bug where a strategy with very few
signals (n=2) and near-identical pnl values produced astronomically
large Sharpe values (e.g. +1.2e15) due to a near-zero variance
denominator.
"""
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import numpy as np
import pytest

from kairos_orchestrator import _safe_sharpe, MIN_SIGNALS_FOR_SHARPE, _SHARPE_CLAMP


class TestSafeSharpe:
    def test_n_below_minimum_returns_zero(self):
        # n=2, identical pnls -> would otherwise divide by ~0 variance.
        rets = np.array([0.01, 0.01])
        assert _safe_sharpe(rets, np.sqrt(252)) == 0.0

    def test_n_equals_min_minus_one_returns_zero(self):
        rets = np.array([0.01] * (MIN_SIGNALS_FOR_SHARPE - 1))
        assert _safe_sharpe(rets, np.sqrt(252)) == 0.0

    def test_near_zero_variance_is_finite_and_clamped(self):
        # Enough samples to pass the n-check, but variance is tiny
        # (not exactly zero) so the naive formula would blow up.
        rets = np.array([0.01, 0.01, 0.01 + 1e-14, 0.01 - 1e-14, 0.01])
        sharpe = _safe_sharpe(rets, np.sqrt(252))
        assert np.isfinite(sharpe)
        assert -_SHARPE_CLAMP <= sharpe <= _SHARPE_CLAMP

    def test_exact_zero_variance_returns_zero(self):
        rets = np.array([0.02, 0.02, 0.02, 0.02])
        assert _safe_sharpe(rets, np.sqrt(252)) == 0.0

    def test_normal_case_unchanged(self):
        rets = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        expected = float(np.mean(rets) / np.std(rets) * np.sqrt(252))
        sharpe = _safe_sharpe(rets, np.sqrt(252))
        assert sharpe == pytest.approx(expected, rel=1e-9)

    def test_pathological_large_returns_clamped(self):
        # Contrived case: large mean relative to a std that survives the
        # epsilon floor but is still tiny -> pre-clamp value would be huge.
        rets = np.array([1.0, 1.0, 1.0 + 1e-12, 1.0 - 1e-12])
        sharpe = _safe_sharpe(rets, np.sqrt(252))
        assert sharpe == pytest.approx(_SHARPE_CLAMP)
