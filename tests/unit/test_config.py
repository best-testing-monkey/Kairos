"""Unit tests for kairos.config (KAI-1)."""
import pytest

import kairos
from kairos.config import _state
from kairos.errors import ConfigError


class TestConfigure:
    def test_defaults_stored(self):
        kairos.configure(remote=False)
        assert _state.calendar == "XNYS"
        assert _state.tz == "America/New_York"

    def test_custom_calendar_stored(self):
        kairos.configure(remote=False, calendar="XLON", tz="Europe/London")
        assert _state.calendar == "XLON"
        assert _state.tz == "Europe/London"
        # reset
        kairos.configure(remote=False)

    def test_bad_calendar_raises_config_error(self):
        with pytest.raises(ConfigError):
            kairos.configure(remote=False, calendar="XBAD_NONEXISTENT")

    def test_no_secrets_stored(self):
        """Kairos state must not hold any API key attributes."""
        kairos.configure(remote=False)
        for attr in dir(_state):
            assert "key" not in attr.lower() and "secret" not in attr.lower()
