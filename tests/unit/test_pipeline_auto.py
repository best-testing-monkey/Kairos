"""Tests for pipeline automation helpers."""

import pytest
from kairos_strategies import _period_to_weeks


class TestPeriodToWeeks:
    """Test the _period_to_weeks period parsing helper."""

    def test_period_to_weeks_values(self):
        """Test period-to-weeks conversion with standard values."""
        # 6m: 6 * (365.25/12) / 7 ≈ 26.089
        assert abs(_period_to_weeks("6m") - 26.09) < 1e-2

        # 1m: 1 * (365.25/12) / 7 ≈ 4.348
        assert abs(_period_to_weeks("1m") - 4.35) < 1e-2

        # 2w: 2 weeks exactly
        assert abs(_period_to_weeks("2w") - 2.0) < 1e-2

        # 1y: 365.25 / 7 ≈ 52.179
        assert abs(_period_to_weeks("1y") - 52.18) < 1e-2

    def test_period_to_weeks_single_unit(self):
        """Test single-unit periods."""
        # 1d: 1 / 7 ≈ 0.143
        assert abs(_period_to_weeks("1d") - 1.0/7.0) < 1e-6

        # 1w: 1 week exactly
        assert _period_to_weeks("1w") == 1.0

        # 1m: 365.25/12/7 ≈ 4.348
        assert abs(_period_to_weeks("1m") - 365.25/12/7) < 1e-6

    def test_period_to_weeks_invalid(self):
        """Test that invalid period strings raise ValueError."""
        # Invalid formats should raise ValueError with the same error type as _period_to_bars
        with pytest.raises(ValueError) as exc_info:
            _period_to_weeks("invalid")
        assert "Unrecognised backtest_period" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            _period_to_weeks("6x")
        assert "Unrecognised backtest_period" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            _period_to_weeks("m")
        assert "Unrecognised backtest_period" in str(exc_info.value)

    def test_period_to_weeks_case_insensitive(self):
        """Test that period strings are case-insensitive."""
        assert _period_to_weeks("6M") == _period_to_weeks("6m")
        assert _period_to_weeks("1Y") == _period_to_weeks("1y")
        assert _period_to_weeks("2W") == _period_to_weeks("2w")

    def test_period_to_weeks_whitespace_tolerant(self):
        """Test that leading/trailing whitespace is handled."""
        assert _period_to_weeks(" 6m ") == _period_to_weeks("6m")
        assert _period_to_weeks("\t1y\t") == _period_to_weeks("1y")
