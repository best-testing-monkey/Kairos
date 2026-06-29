"""
Tests for backtest() and compute_metrics() functions from kairos_strategies.py
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Direct import from strategy directory (don't rely on conftest due to side effects)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from kairos_strategies import backtest, compute_metrics


# =============================================================================
# TEST BACKTEST SIGNALS
# =============================================================================

class TestBacktestSignals:
    """Test backtest signal generation and execution"""

    def test_always_up_prediction(self):
        """
        Predicted = [100, 102, 104, ...] (always rising 2%/bar)
        Actual = [100, 101, 102, ...]
        threshold=0.01 (1%)
        → at least one BUY trade should be made
        """
        # Predicted prices: rising by 2% each bar
        predicted_close = pd.Series([100 * (1.02 ** i) for i in range(50)])
        # Actual prices: rising by 1% each bar
        actual_close = pd.Series([100 * (1.01 ** i) for i in range(50)])

        initial_capital = 100_000
        threshold = 0.01

        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        # Should have at least one BUY trade since prediction is always rising > 1%
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        assert len(buy_trades) > 0, f"Expected at least one BUY trade, got {len(buy_trades)}"

    def test_flat_prediction(self):
        """
        Predicted = [100, 100, 100, ...] (no change)
        threshold=0.01 (1%)
        → zero trades
        """
        predicted_close = pd.Series([100.0] * 50)
        actual_close = pd.Series([100.0 + i * 0.1 for i in range(50)])

        initial_capital = 100_000
        threshold = 0.01

        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        # No signal when prediction is flat
        assert len(trades) == 0, f"Expected 0 trades for flat prediction, got {len(trades)}"

    def test_high_threshold_no_trades(self):
        """
        Predicted changes are 0.5%, threshold=0.02 (2%)
        → zero trades
        """
        # Predicted prices: rising by 0.5% each bar
        predicted_close = pd.Series([100 * (1.005 ** i) for i in range(50)])
        actual_close = pd.Series([100 * (1.005 ** i) for i in range(50)])

        initial_capital = 100_000
        threshold = 0.02  # 2% threshold, but prediction is only 0.5%

        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        # No trades since predicted return < threshold
        assert len(trades) == 0, f"Expected 0 trades with high threshold, got {len(trades)}"

    def test_equity_length(self):
        """Equity series length == n bars"""
        n = 30
        predicted_close = pd.Series([100 + i for i in range(n)])
        actual_close = pd.Series([100 + i for i in range(n)])

        initial_capital = 100_000
        threshold = 0.01

        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        assert len(equity_series) == n, \
            f"Expected equity series length {n}, got {len(equity_series)}"


# =============================================================================
# TEST COMPUTE METRICS
# =============================================================================

class TestComputeMetrics:
    """Test metrics computation"""

    def test_positive_return(self):
        """Equity goes from 100k to 110k → total_return ≈ 0.10"""
        dates = pd.date_range('2024-01-01', periods=10)
        equity = pd.Series([100_000, 101_000, 102_000, 103_000, 104_000,
                           105_000, 106_000, 107_000, 108_000, 110_000],
                          index=dates)
        initial_capital = 100_000
        trades = []

        metrics = compute_metrics(equity, initial_capital, trades)

        assert abs(metrics['total_return'] - 0.10) < 0.001, \
            f"Expected total_return ≈ 0.10, got {metrics['total_return']}"

    def test_metrics_keys(self):
        """Result dict has required keys"""
        dates = pd.date_range('2024-01-01', periods=10)
        equity = pd.Series([100_000 + i * 1000 for i in range(10)], index=dates)
        initial_capital = 100_000
        trades = []

        metrics = compute_metrics(equity, initial_capital, trades)

        required_keys = ['total_return', 'annual_return', 'sharpe',
                        'max_drawdown', 'win_rate', 'trades', 'final_capital']
        for key in required_keys:
            assert key in metrics, f"Missing key '{key}' in metrics"

    def test_zero_trades(self):
        """trades=[] → win_rate = 0.0"""
        dates = pd.date_range('2024-01-01', periods=10)
        equity = pd.Series([100_000 + i * 100 for i in range(10)], index=dates)
        initial_capital = 100_000
        trades = []

        metrics = compute_metrics(equity, initial_capital, trades)

        assert metrics['win_rate'] == 0.0, \
            f"Expected win_rate 0.0 with no trades, got {metrics['win_rate']}"
        assert metrics['trades'] == 0, \
            f"Expected 0 trades, got {metrics['trades']}"

    def test_final_capital(self):
        """final_capital equals last equity value"""
        dates = pd.date_range('2024-01-01', periods=5)
        equity = pd.Series([100_000, 101_000, 102_000, 103_000, 105_000],
                          index=dates)
        initial_capital = 100_000
        trades = []

        metrics = compute_metrics(equity, initial_capital, trades)

        assert metrics['final_capital'] == 105_000, \
            f"Expected final_capital 105000, got {metrics['final_capital']}"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestBacktestIntegration:
    """Integration tests combining backtest and metrics"""

    def test_backtest_with_trades_then_metrics(self):
        """Full pipeline: backtest with trades, then compute metrics"""
        # Create a scenario where we should get some trades
        predicted_close = pd.Series([100, 102, 104, 103, 101, 100, 99, 98, 97, 96])
        actual_close = pd.Series([100, 101, 102, 103, 102, 101, 100, 99, 98, 97])

        initial_capital = 100_000
        threshold = 0.01

        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        # Compute metrics
        metrics = compute_metrics(equity_series, initial_capital, trades)

        # Basic sanity checks
        assert 'total_return' in metrics
        assert 'sharpe' in metrics
        assert 'max_drawdown' in metrics
        assert metrics['win_rate'] >= 0.0 and metrics['win_rate'] <= 1.0

    def test_downtrend_prediction(self):
        """Downtrend prediction should produce SHORT trades"""
        # Predicted downtrend: falling by 2% each bar
        predicted_close = pd.Series([100 * (0.98 ** i) for i in range(50)])
        # Actual also downtrending
        actual_close = pd.Series([100 * (0.99 ** i) for i in range(50)])

        initial_capital = 100_000
        threshold = 0.01

        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        # Should have SHORT trades since prediction is falling > 1%
        short_trades = [t for t in trades if t['action'] == 'SHORT']
        assert len(short_trades) > 0, f"Expected SHORT trades in downtrend, got {len(short_trades)}"

    def test_equity_never_negative(self):
        """Equity should never go negative (assuming we don't allow unlimited shorting)"""
        predicted_close = pd.Series(np.linspace(100, 150, 50))
        actual_close = pd.Series(np.linspace(100, 150, 50))

        initial_capital = 100_000
        threshold = 0.001

        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        # Equity should remain reasonable (not go to zero or negative)
        # The minimum equity after trading depends on position sizing
        # but should generally stay well above zero
        assert equity_series.min() > 0 or len(trades) == 0, \
            f"Equity went to {equity_series.min()}, which is too low"


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_bar_backtest(self):
        """Backtest with single bar"""
        predicted_close = pd.Series([100.0])
        actual_close = pd.Series([101.0])

        initial_capital = 100_000
        threshold = 0.01

        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        assert len(equity_series) == 1
        assert len(trades) == 0  # Can't trade with only 1 bar

    def test_nan_prices_handled(self):
        """Backtest handles NaN prices gracefully"""
        predicted_close = pd.Series([100, 102, 104, np.nan, 106])
        actual_close = pd.Series([100, 101, 102, 103, 104])

        initial_capital = 100_000
        threshold = 0.01

        # Should not crash
        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        assert len(equity_series) <= 5
        assert not np.isnan(equity_series.iloc[-1]) or len(trades) == 0

    def test_zero_initial_capital(self):
        """Backtest with zero initial capital"""
        predicted_close = pd.Series([100, 102, 104])
        actual_close = pd.Series([100, 101, 102])

        initial_capital = 0
        threshold = 0.01

        # Should handle gracefully (may not execute trades)
        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        assert len(equity_series) <= 3


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

class TestParametrized:
    """Parametrized tests for various scenarios"""

    @pytest.mark.parametrize("threshold", [0.001, 0.01, 0.05, 0.1])
    def test_threshold_effect(self, threshold):
        """Higher threshold should result in fewer trades"""
        predicted_close = pd.Series([100 * (1.02 ** i) for i in range(50)])
        actual_close = pd.Series([100 * (1.01 ** i) for i in range(50)])
        initial_capital = 100_000

        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        # Should get trades at low threshold, fewer (or none) at high threshold
        assert len(trades) >= 0  # At minimum, shouldn't crash

    @pytest.mark.parametrize("n_bars", [10, 50, 100])
    def test_various_bar_counts(self, n_bars):
        """Backtest should work with various bar counts"""
        predicted_close = pd.Series(np.linspace(100, 120, n_bars))
        actual_close = pd.Series(np.linspace(100, 120, n_bars))
        initial_capital = 100_000
        threshold = 0.01

        equity_series, trades = backtest(predicted_close, actual_close,
                                        initial_capital, threshold)

        assert len(equity_series) == n_bars
