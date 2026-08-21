import importlib
import sys
import os
from unittest.mock import MagicMock

import pytest

# Allow tests to import from strategy/ and scripts/ directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "strategy"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

# Every module that imports send_telegram/send_telegram_document by name from
# kairos.ops (a "from X import Y" import binds the name into the importing
# module's own namespace, so each one needs patching separately -- patching
# kairos.ops.send_telegram alone would NOT affect code that already did
# `from kairos.ops import send_telegram`).
_TELEGRAM_IMPORTING_MODULES = (
    "kairos_gpu", "kairos_papertrade", "kairos_pipeline",
    "kairos_daily_signals", "kairos_weekly_discovery",
)


@pytest.fixture(autouse=True)
def _no_real_telegram_sends(monkeypatch):
    """Safety net: a unit test must never send a real Telegram message,
    regardless of whether the specific test remembered to mock send_telegram
    locally.

    Found 2026-08-21: several tests called real production notification code
    paths without mocking send_telegram -- test_gpu_recover.py's
    test_recovery_invoked_with_correct_resume_cmd/test_allow_reboot_env_passed_through/
    test_recovery_failure_raises_runtime_error called the real
    kairos_gpu.ensure_cuda() (whose GPU-recovery notifications are
    intentionally *never* gated by any enable flag -- that's by design, see
    kairos_gpu.py, not a bug to "fix" there), and most of
    test_kairos_papertrade.py's TestPrewarmPredictionCache tests called the
    real kairos_papertrade.prewarm_prediction_cache() with its notify
    parameter left at its default (True). In a sandboxed CI-style
    environment with no TELEGRAM_BOT_TOKEN set, send_telegram's OpsError
    was silently swallowed and logged as a warning -- but on a real dev
    machine with credentials already in the shell, every test run fired
    real messages to a real phone.

    This patches send_telegram (and send_telegram_document) to a no-op Mock
    in every module that imports either by name, for every test, by
    default. A test that wants to verify actual notify call args/content
    should locally `monkeypatch.setattr(some_module, "send_telegram",
    its_own_mock)` inside the test body -- that still works exactly as
    before, since a later setattr always overrides this fixture's earlier
    one within the same test.
    """
    noop = MagicMock()
    for modname in _TELEGRAM_IMPORTING_MODULES:
        mod = sys.modules.get(modname)
        if mod is None:
            try:
                mod = importlib.import_module(modname)
            except ImportError:
                continue
        monkeypatch.setattr(mod, "send_telegram", noop, raising=False)
        monkeypatch.setattr(mod, "send_telegram_document", noop, raising=False)
