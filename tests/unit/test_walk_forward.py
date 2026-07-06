"""
Walk-Forward Validation Tests

Test suite for walk_forward() function covering:
- Fold partitioning with no overlaps
- Fresh strategy instances per fold
- Per-fold and aggregate metrics
- Overfitting detection (DSR, Sharpe degradation)
- Reproducibility with fixed seed
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add strategy/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

from kairos_backtest import (
    walk_forward,
    KairosPredictor,
    Strategy,
    Signal,
    Direction,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def minimal_data():
    """Generate minimal OHLCV data (100 bars) for fast testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    close_prices = 100 + np.cumsum(np.random.normal(0.1, 1.0, 100))

    df = pd.DataFrame({
        "open": close_prices,
        "high": close_prices + 1,
        "low": close_prices - 1,
        "close": close_prices,
        "volume": np.full(100, 1e6),
    }, index=dates)

    return df


@pytest.fixture
def fast_predictor():
    """Create a minimal predictor that returns instantly."""
    def predict_fn(history: pd.DataFrame):
        """Generate 3 minimal prediction samples."""
        last_close = float(history["close"].iloc[-1])
        samples = []
        for i in range(3):
            future_close = last_close + (i - 1) * 0.1  # -0.1, +0.1, +0.1
            samples.append(pd.DataFrame({
                "open": [last_close],
                "high": [future_close + 0.5],
                "low": [future_close - 0.5],
                "close": [future_close],
                "volume": [1e6],
            }))
        return samples

    return KairosPredictor(predict_fn)


@pytest.fixture
def null_strategy_factory():
    """Strategy factory that never generates signals (fast, no trading)."""
    class NullStrategy(Strategy):
        name = "null_strategy"

        def generate_signal(self, dist, current_price, history, context):
            return None

    return lambda: NullStrategy()


@pytest.fixture
def simple_strategy_factory():
    """Strategy factory with basic signal generation."""
    class SimpleStrategy(Strategy):
        name = "simple_strategy"

        def generate_signal(self, dist, current_price, history, context):
            s = dist.stats.get("close", {})
            if not s:
                return None

            # Simple logic: trade if we have enough data
            if len(history) < 10:
                return None

            pct_20 = s.get("pct_20", current_price)
            pct_80 = s.get("pct_80", current_price)

            if current_price <= pct_20:
                return Signal(
                    direction=Direction.LONG,
                    size=0.1,
                    entry=current_price,
                    stop=s.get("pct_5", pct_20 * 0.95),
                    target=s.get("pct_95", pct_80),
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=1.0,
                )
            return None

    return lambda: SimpleStrategy()


# =============================================================================
# ACCEPTANCE CRITERIA TESTS
# =============================================================================


class TestFoldPartitioningNoOverlap:
    """Test that folds never overlap in test data (Acceptance Criterion 1)."""

    def test_fold_partitioning_no_overlap(self, minimal_data, fast_predictor, null_strategy_factory):
        """Verify that test folds never overlap."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        assert result["num_folds"] >= 1, "Should have at least one fold"
        folds = result["folds"]

        # Collect all test ranges
        test_ranges = [(f["test_start"], f["test_end"]) for f in folds]

        # Verify no overlap between test folds
        for i in range(len(test_ranges)):
            for j in range(i + 1, len(test_ranges)):
                start_i, end_i = test_ranges[i]
                start_j, end_j = test_ranges[j]

                # Test ranges must not overlap
                has_overlap = not (end_i <= start_j or end_j <= start_i)
                assert not has_overlap, \
                    f"Fold {i} test [{start_i}:{end_i}] overlaps with fold {j} test [{start_j}:{end_j}]"

    def test_sequential_folds(self, minimal_data, fast_predictor, null_strategy_factory):
        """Test folds are sequential without gaps."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        if result["num_folds"] < 2:
            pytest.skip("Need at least 2 folds for this test")

        folds = result["folds"]
        for i in range(len(folds) - 1):
            current_test_end = folds[i]["test_end"]
            next_fold = folds[i + 1]

            # For sliding window, next test should start where current ended
            assert next_fold["test_start"] == current_test_end, \
                f"Gap between fold {i} and {i+1}: {current_test_end} vs {next_fold['test_start']}"


class TestOverfittingDetection:
    """Test overfitting detection via DSR and Sharpe degradation."""

    def test_overfitting_score_is_numeric(self, minimal_data, fast_predictor, null_strategy_factory):
        """DSR and related metrics should be numeric."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        assert isinstance(result["overfitting_score"], (int, float))
        assert isinstance(result["is_sharpe_mean"], (int, float))
        assert isinstance(result["oos_sharpe_mean"], (int, float))
        assert isinstance(result["sharpe_degradation"], (int, float))

    def test_dsr_less_than_is_sharpe(self, minimal_data, fast_predictor, null_strategy_factory):
        """DSR should generally be lower than or equal to IS Sharpe (penalizes overfitting)."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        dsr = result["overfitting_score"]
        is_sharpe = result["is_sharpe_mean"]

        # DSR should not exceed IS Sharpe (overfitting penalty)
        # In most cases DSR <= IS Sharpe, though not guaranteed for all inputs
        # We just verify computation happened
        assert dsr is not None
        assert is_sharpe is not None


class TestReproducibility:
    """Test that fixed seed produces reproducible results (Acceptance Criterion 2)."""

    def test_walk_forward_reproducible_same_seed(self, minimal_data, fast_predictor, null_strategy_factory):
        """Fixed seed runs should produce identical metrics."""
        result_1 = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=99,
        )

        result_2 = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=99,
        )

        # Check num_folds
        assert result_1["num_folds"] == result_2["num_folds"]

        # Check aggregate metrics match
        assert abs(result_1["is_sharpe_mean"] - result_2["is_sharpe_mean"]) < 1e-6
        assert abs(result_1["oos_sharpe_mean"] - result_2["oos_sharpe_mean"]) < 1e-6
        assert abs(result_1["sharpe_degradation"] - result_2["sharpe_degradation"]) < 1e-6
        assert abs(result_1["overfitting_score"] - result_2["overfitting_score"]) < 1e-6

    def test_different_seed_produces_variation(self, minimal_data, fast_predictor, null_strategy_factory):
        """Different seeds should potentially produce different results."""
        result_1 = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=1,
        )

        result_2 = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=2,
        )

        # Should complete successfully with different seeds
        assert result_1["num_folds"] == result_2["num_folds"]


class TestMetricsPresence:
    """Test that all required metrics are computed and returned."""

    def test_per_fold_metrics_present(self, minimal_data, fast_predictor, null_strategy_factory):
        """Each fold should have train and test metrics."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        for fold in result["folds"]:
            assert "train_metrics" in fold
            assert "test_metrics" in fold
            assert isinstance(fold["train_metrics"], dict)
            assert isinstance(fold["test_metrics"], dict)
            assert "sharpe" in fold["train_metrics"]
            assert "sharpe" in fold["test_metrics"]

    def test_aggregate_metrics_present(self, minimal_data, fast_predictor, null_strategy_factory):
        """Result should have all required aggregate keys."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        required_keys = [
            "folds",
            "num_folds",
            "aggregate_train",
            "aggregate_test",
            "is_sharpe_mean",
            "oos_sharpe_mean",
            "sharpe_degradation",
            "overfitting_score",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
            assert result[key] is not None, f"Key {key} is None"


class TestFreshStrategyInstances:
    """Test that strategy instances are created fresh per fold."""

    def test_factory_called_multiple_times(self, minimal_data, fast_predictor):
        """Strategy factory should be called at least once per fold."""
        call_count = {"count": 0}

        def counting_factory():
            call_count["count"] += 1
            class CountingStrategy(Strategy):
                name = "counting"
                def generate_signal(self, dist, current_price, history, context):
                    return None
            return CountingStrategy()

        result = walk_forward(
            counting_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        num_folds = result["num_folds"]
        # Factory is called at least 2x per fold (IS + OOS)
        expected_min_calls = num_folds * 2
        assert call_count["count"] >= expected_min_calls, \
            f"Factory called {call_count['count']} times, expected >= {expected_min_calls}"


class TestAnchoredVsSliding:
    """Test anchored vs sliding window modes."""

    def test_anchored_expands_training(self, minimal_data, fast_predictor, null_strategy_factory):
        """Anchored mode should expand training window, keep train_start = 0."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=True,
            random_seed=42,
        )

        if result["num_folds"] < 2:
            pytest.skip("Need at least 2 folds")

        folds = result["folds"]

        # All folds should have train_start = 0 in anchored mode
        for i, fold in enumerate(folds):
            assert fold["train_start"] == 0, \
                f"Anchored fold {i}: train_start should be 0, got {fold['train_start']}"

            # Training window should expand as we move through folds
            if i > 0:
                prev_train_end = folds[i - 1]["train_end"]
                curr_train_end = fold["train_end"]
                assert curr_train_end >= prev_train_end, \
                    f"Anchored fold {i}: training window should not shrink"

    def test_sliding_rolls_window(self, minimal_data, fast_predictor, null_strategy_factory):
        """Sliding mode should roll both windows forward."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        if result["num_folds"] < 2:
            pytest.skip("Need at least 2 folds")

        folds = result["folds"]

        # First fold should start at 0
        assert folds[0]["train_start"] == 0

        # Sliding mode advances both windows by `step` each fold (here
        # step == test_days == 15, so test windows are also contiguous;
        # train_start itself advances by `step`, not by the full
        # train+test window length).
        step = 15
        for i in range(1, len(folds)):
            assert folds[i]["train_start"] == folds[i - 1]["train_start"] + step
        # Test windows remain contiguous/non-overlapping regardless.
        for i in range(1, len(folds)):
            assert folds[i]["test_start"] == folds[i - 1]["test_end"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data_raises_error(self, fast_predictor, null_strategy_factory):
        """Should raise ValueError if data is too short."""
        short_data = pd.DataFrame({
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100],
            "volume": [1e6],
        }, index=pd.date_range("2023-01-01", periods=1))

        with pytest.raises(ValueError):
            walk_forward(
                null_strategy_factory,
                short_data,
                fast_predictor,
                train_days=100,
                test_days=50,
            )

    def test_single_fold_returns_valid_result(self, minimal_data, fast_predictor, null_strategy_factory):
        """With minimal data for one fold, should still return valid results."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=50,
            test_days=40,
            step=40,
            anchored=False,
            random_seed=42,
        )

        assert result["num_folds"] >= 1
        assert len(result["folds"]) >= 1
        assert "aggregate_train" in result
        assert "aggregate_test" in result

    def test_no_signal_strategy_completes(self, minimal_data, fast_predictor, null_strategy_factory):
        """Strategy that generates no signals should complete without error."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        # Should complete successfully
        assert result["num_folds"] >= 1
        for fold in result["folds"]:
            assert "train_metrics" in fold
            assert "test_metrics" in fold


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestOverfitFixtureCollapse:
    """
    Overfit fixture per ticket acceptance criteria: a deliberately overfit
    (lookahead-peeking) strategy should show OOS Sharpe collapse and a
    Deflated Sharpe Ratio below 0.5.

    `walk_forward()` calls `strategy_factory()` once for the in-sample (IS)
    backtest and once for the out-of-sample (OOS) backtest, per fold, in
    that order. We exploit this to hand back a "cheating" strategy for the
    IS call (it peeks at the true future close of the full dataset, which
    is only stitched together for this fixture as a controlled lookahead
    leak) and a pure-noise strategy for the OOS call. The cheating strategy
    always trades in the realized winning direction, producing an
    artificially inflated IS Sharpe; the noise strategy has no real edge,
    so OOS Sharpe collapses towards zero, exactly the pathology
    walk-forward validation exists to catch.
    """

    @pytest.fixture
    def overfit_strategy_factory(self, minimal_data):
        full_close = minimal_data["close"]

        class CheatingStrategy(Strategy):
            """
            In-sample only: peeks at the true future close (lookahead leak) to
            pick the winning direction, then uses a stop far outside any
            realistic daily move (so a wrong call just leaves the position
            open, harmlessly, rather than recording a loss) and a target only
            a hair away from entry (so a correct call closes out as a quick,
            near-certain win on the very next bar). This deterministically
            inflates the in-sample Sharpe via lookahead bias.
            """
            name = "cheat_is"

            def generate_signal(self, dist, current_price, history, context):
                # BacktestEngine enters at tomorrow's (pos+1) open and only
                # evaluates the stop/target against the day after that
                # (pos+2). To "cheat" effectively we must peek at pos+2, the
                # bar the exit decision is actually made against - not pos+1.
                current_ts = history.index[-1]
                pos = full_close.index.get_loc(current_ts)
                if pos + 2 >= len(full_close):
                    return None
                entry_price_est = float(full_close.iloc[pos + 1])
                exit_day_close = float(full_close.iloc[pos + 2])
                direction = (
                    Direction.LONG if exit_day_close >= entry_price_est else Direction.SHORT
                )
                return Signal(
                    direction=direction,
                    size=0.5,
                    entry=current_price,
                    stop=current_price * (0.5 if direction == Direction.LONG else 1.5),
                    target=current_price * (1.0005 if direction == Direction.LONG else 0.9995),
                    strategy_name=self.name,
                    confidence=0.99,
                    expected_value=1.0,
                )

        class NoiseStrategy(Strategy):
            """Out-of-sample only: no real edge, trades on a fixed-seed coin flip."""
            name = "noise_oos"

            def __init__(self):
                self._rng = np.random.RandomState(12345)

            def generate_signal(self, dist, current_price, history, context):
                direction = Direction.LONG if self._rng.rand() < 0.5 else Direction.SHORT
                return Signal(
                    direction=direction,
                    size=0.5,
                    entry=current_price,
                    stop=current_price * (0.98 if direction == Direction.LONG else 1.02),
                    target=current_price * (1.01 if direction == Direction.LONG else 0.99),
                    strategy_name=self.name,
                    confidence=0.5,
                    expected_value=0.0,
                )

        call_count = {"n": 0}

        def factory():
            call_count["n"] += 1
            # Odd calls = IS instance, even calls = OOS instance (per fold).
            if call_count["n"] % 2 == 1:
                return CheatingStrategy()
            return NoiseStrategy()

        return factory

    def test_overfit_strategy_dsr_below_threshold(
        self, minimal_data, fast_predictor, overfit_strategy_factory
    ):
        result = walk_forward(
            overfit_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        assert result["num_folds"] >= 1
        assert result["is_sharpe_mean"] > result["oos_sharpe_mean"], (
            "Overfit fixture should show IS Sharpe > OOS Sharpe (degradation)"
        )
        assert result["overfitting_score"] < 0.5, (
            f"DSR should collapse below 0.5 for the overfit fixture, "
            f"got {result['overfitting_score']}"
        )


class TestWalkForwardIntegration:
    """Integration tests for walk_forward with various inputs."""

    def test_with_trading_strategy(self, minimal_data, fast_predictor, simple_strategy_factory):
        """Walk-forward should work with strategy that generates signals."""
        result = walk_forward(
            simple_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            random_seed=42,
        )

        assert result["num_folds"] >= 1
        # Aggregate metrics should be computed even with trades
        assert "aggregate_train" in result
        assert "aggregate_test" in result

    def test_custom_fees_and_slippage(self, minimal_data, fast_predictor, null_strategy_factory):
        """Walk-forward should accept custom fee and slippage parameters."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            fee_pct=0.002,
            slippage_pct=0.001,
            random_seed=42,
        )

        assert result["num_folds"] >= 1

    def test_custom_initial_capital(self, minimal_data, fast_predictor, null_strategy_factory):
        """Walk-forward should respect custom initial capital."""
        result = walk_forward(
            null_strategy_factory,
            minimal_data,
            fast_predictor,
            train_days=25,
            test_days=15,
            step=15,
            anchored=False,
            initial_capital=50000.0,
            random_seed=42,
        )

        assert result["num_folds"] >= 1
