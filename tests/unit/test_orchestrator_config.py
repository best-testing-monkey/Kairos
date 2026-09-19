import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
from dataclasses import replace
from kairos_orchestrator import (
    OrchestratorConfig, _FILTER_PRESETS_BY_INTERVAL,
    PREDICTION_USAGE_MODES,
)
from kairos_meta import AssetPrediction
from kairos_backtest import KairosDistribution
import pandas as pd
import numpy as np


class TestOrchestratorConfigForInterval:
    """Test OrchestratorConfig.for_interval classmethod and preset fallback."""

    def test_default_interval_matches_dataclass_defaults(self):
        """Verify that for_interval('1d') produces same values as OrchestratorConfig()."""
        default_config = OrchestratorConfig()
        from_method = OrchestratorConfig.for_interval("1d")

        assert from_method.entropy_threshold == default_config.entropy_threshold == 3.0
        assert from_method.kurtosis_max == default_config.kurtosis_max == 10.0
        assert from_method.min_volume_percentile == default_config.min_volume_percentile == 10.0

    def test_uncalibrated_interval_falls_back_to_defaults(self):
        """Verify that any uncalibrated interval falls back to dataclass defaults."""
        default_config = OrchestratorConfig()
        from_method = OrchestratorConfig.for_interval("1h")

        assert from_method.entropy_threshold == default_config.entropy_threshold
        assert from_method.kurtosis_max == default_config.kurtosis_max
        assert from_method.min_volume_percentile == default_config.min_volume_percentile

    def test_preset_lookup_and_merge(self):
        """Verify that presets are correctly looked up and merged."""
        # Monkeypatch a test preset for __test_only__
        original = _FILTER_PRESETS_BY_INTERVAL.get("__test_only__")
        try:
            _FILTER_PRESETS_BY_INTERVAL["__test_only__"] = {
                "entropy_threshold": 1.5,
            }
            config = OrchestratorConfig.for_interval("__test_only__")
            assert config.entropy_threshold == 1.5
            # Other fields should stay at dataclass defaults
            assert config.kurtosis_max == 10.0
            assert config.min_volume_percentile == 10.0
        finally:
            # Cleanup
            if original is None:
                _FILTER_PRESETS_BY_INTERVAL.pop("__test_only__", None)
            else:
                _FILTER_PRESETS_BY_INTERVAL["__test_only__"] = original

    def test_explicit_overrides_win_over_presets(self):
        """Verify that explicit **overrides take precedence over presets."""
        original = _FILTER_PRESETS_BY_INTERVAL.get("__test_only__")
        try:
            _FILTER_PRESETS_BY_INTERVAL["__test_only__"] = {
                "entropy_threshold": 1.5,
                "kurtosis_max": 8.0,
            }
            config = OrchestratorConfig.for_interval(
                "__test_only__",
                entropy_threshold=2.0,
                disabled_strategies={"foo"},
            )
            # Override should win
            assert config.entropy_threshold == 2.0
            # Preset should be used for kurtosis_max
            assert config.kurtosis_max == 8.0
            # Override should win for disabled_strategies
            assert config.disabled_strategies == {"foo"}
        finally:
            if original is None:
                _FILTER_PRESETS_BY_INTERVAL.pop("__test_only__", None)
            else:
                _FILTER_PRESETS_BY_INTERVAL["__test_only__"] = original

    def test_disabled_strategies_override_works(self):
        """Verify explicit overrides still work alongside presets."""
        config = OrchestratorConfig.for_interval("1d", disabled_strategies={"foo"})
        assert config.disabled_strategies == {"foo"}


class TestOrchestratorConfigPredictionUsageMode:
    """Test OrchestratorConfig.prediction_usage_mode field."""

    def test_prediction_usage_mode_default_value(self):
        """Verify that prediction_usage_mode defaults to 'last_real_bar'."""
        config = OrchestratorConfig()
        assert config.prediction_usage_mode == "last_real_bar"

    def test_prediction_usage_mode_constructor_kwarg(self):
        """Verify that prediction_usage_mode can be set via constructor kwarg."""
        config = OrchestratorConfig(prediction_usage_mode="distribution_as_bar")
        assert config.prediction_usage_mode == "distribution_as_bar"


class TestPredictionUsageModes:
    """Test PREDICTION_USAGE_MODES registry and dispatch."""

    def test_registry_identity_mode_exists(self):
        """Verify that the identity mode 'last_real_bar' is in the registry."""
        assert "last_real_bar" in PREDICTION_USAGE_MODES
        assert callable(PREDICTION_USAGE_MODES["last_real_bar"])

    def test_identity_mode_returns_unchanged_copy(self):
        """Verify that 'last_real_bar' mode returns an unchanged copy via replace()."""
        # Create a minimal test AssetPrediction using mocked dist
        hist = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
            "open": [99.0, 100.0, 101.0],
            "high": [102.0, 103.0, 104.0],
            "low": [98.0, 99.0, 100.0],
        })

        # Create a minimal KairosDistribution by passing a list of DataFrames
        dist_samples = [
            pd.DataFrame({"open": [101.0], "high": [104.0], "low": [100.0], "close": [102.0]}),
            pd.DataFrame({"open": [101.5], "high": [104.5], "low": [100.5], "close": [102.5]}),
        ]
        dist = KairosDistribution(dist_samples)

        pred = AssetPrediction(
            symbol="BTC-USD",
            dist=dist,
            current_price=102.0,
            history=hist,
        )

        # Apply the identity transform
        identity_fn = PREDICTION_USAGE_MODES["last_real_bar"]
        result = identity_fn(pred)

        # Verify the result is equal (but not the same object)
        assert result is not pred  # Different objects (via replace())
        assert result.symbol == pred.symbol
        assert result.current_price == pred.current_price
        assert result.dist is pred.dist
        assert result.history is pred.history

    def test_unknown_mode_raises_keyerror(self):
        """Verify that an unknown prediction_usage_mode raises KeyError at dispatch time."""
        # Trying to look up an invalid mode should raise KeyError
        with pytest.raises(KeyError):
            PREDICTION_USAGE_MODES["nonexistent_mode"]

    def test_custom_mode_via_monkeypatch(self):
        """Verify that a custom mode function can be registered and applied via monkeypatch."""
        # Create a test prediction
        hist = pd.DataFrame({"close": [100.0, 101.0]})
        dist_samples = [
            pd.DataFrame({"open": [100.0], "high": [102.0], "low": [99.0], "close": [100.5]}),
        ]
        dist = KairosDistribution(dist_samples)

        pred = AssetPrediction(
            symbol="BTC-USD",
            dist=dist,
            current_price=101.0,
            history=hist,
        )

        # Track whether the custom function was called
        call_log = []

        def custom_transform(p: AssetPrediction) -> AssetPrediction:
            """Custom transform that modifies current_price and logs the call."""
            call_log.append(p.symbol)
            # Return a modified copy with current_price doubled
            return replace(p, current_price=p.current_price * 2.0)

        # Register the custom mode
        original = PREDICTION_USAGE_MODES.get("test_custom")
        try:
            PREDICTION_USAGE_MODES["test_custom"] = custom_transform

            # Apply the custom transform
            result = PREDICTION_USAGE_MODES["test_custom"](pred)

            # Verify the transform was applied
            assert result.current_price == 202.0
            assert "BTC-USD" in call_log
        finally:
            # Cleanup
            if original is None:
                PREDICTION_USAGE_MODES.pop("test_custom", None)
            else:
                PREDICTION_USAGE_MODES["test_custom"] = original

    def test_multiple_preds_transformed_independently(self):
        """Verify that the dispatch loop correctly transforms multiple predictions."""
        # Create two test predictions
        hist1 = pd.DataFrame({"close": [100.0]})
        dist1_samples = [
            pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5]}),
        ]
        dist1 = KairosDistribution(dist1_samples)

        pred1 = AssetPrediction(
            symbol="BTC-USD",
            dist=dist1,
            current_price=100.0,
            history=hist1,
        )

        hist2 = pd.DataFrame({"close": [50.0]})
        dist2_samples = [
            pd.DataFrame({"open": [50.0], "high": [51.0], "low": [49.0], "close": [50.5]}),
        ]
        dist2 = KairosDistribution(dist2_samples)

        pred2 = AssetPrediction(
            symbol="ETH-USD",
            dist=dist2,
            current_price=50.0,
            history=hist2,
        )

        multi_preds = {"BTC-USD": pred1, "ETH-USD": pred2}

        # Apply the identity transform to all
        identity_fn = PREDICTION_USAGE_MODES["last_real_bar"]
        transformed = {sym: identity_fn(pred) for sym, pred in multi_preds.items()}

        # Verify both were transformed and are unchanged (identity mode)
        assert len(transformed) == 2
        assert transformed["BTC-USD"].current_price == 100.0
        assert transformed["ETH-USD"].current_price == 50.0
        assert transformed["BTC-USD"].symbol == "BTC-USD"
        assert transformed["ETH-USD"].symbol == "ETH-USD"
