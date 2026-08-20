import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))

import pytest
from kairos_orchestrator import OrchestratorConfig, _FILTER_PRESETS_BY_INTERVAL


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
