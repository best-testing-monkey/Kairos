"""Unit tests for scripts/kairos_idle_finetune.py: idle detection and command building.

No GPU or network access is required - nvidia-smi/GpuLock/subprocess calls are
either exercised against real local files (the lock probe) or fully replaced
with injected fakes (utilization sampling, timing).
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "kairos_idle_finetune", REPO_ROOT / "scripts" / "kairos_idle_finetune.py"
)
idle_finetune = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(idle_finetune)


@pytest.fixture(autouse=True)
def _isolate_state_file(monkeypatch, tmp_path):
    """Keep tests from reading/writing the real idle finetune state file."""
    monkeypatch.setattr(idle_finetune, "STATE_FILE", tmp_path / "idle_finetune_state.json")


# ---------------------------------------------------------------------------
# is_gpu_idle
# ---------------------------------------------------------------------------

def test_is_gpu_idle_all_samples_below_threshold():
    sleeps = []
    assert idle_finetune.is_gpu_idle(
        threshold=10,
        samples=3,
        interval_seconds=5,
        util_fn=lambda: 2,
        sleep_fn=lambda s: sleeps.append(s),
    )
    # Slept between samples, not after the last one.
    assert sleeps == [5, 5]


def test_is_gpu_idle_returns_false_when_busy():
    calls = {"n": 0}

    def util_fn():
        calls["n"] += 1
        return 50

    sleeps = []
    assert not idle_finetune.is_gpu_idle(
        threshold=10, samples=3, interval_seconds=5, util_fn=util_fn, sleep_fn=lambda s: sleeps.append(s)
    )
    # Returns on the first over-threshold sample, no need to keep sampling.
    assert calls["n"] == 1
    assert sleeps == []


def test_is_gpu_idle_returns_false_partway_through():
    values = iter([1, 2, 99])
    assert not idle_finetune.is_gpu_idle(
        threshold=10, samples=3, interval_seconds=1, util_fn=lambda: next(values), sleep_fn=lambda s: None
    )


def test_is_gpu_idle_treats_missing_reading_as_not_idle():
    assert not idle_finetune.is_gpu_idle(
        threshold=10, samples=3, interval_seconds=1, util_fn=lambda: None, sleep_fn=lambda s: None
    )


def test_is_gpu_idle_at_threshold_is_not_idle():
    # >= threshold counts as busy, not idle.
    assert not idle_finetune.is_gpu_idle(
        threshold=10, samples=1, interval_seconds=1, util_fn=lambda: 10, sleep_fn=lambda s: None
    )


# ---------------------------------------------------------------------------
# lock_is_available
# ---------------------------------------------------------------------------

def test_lock_is_available_when_free(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    assert idle_finetune.lock_is_available(lock_path)


def test_lock_is_available_false_when_held(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not idle_finetune.lock_is_available(lock_path)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_lock_is_available_again_after_release(tmp_path):
    lock_path = tmp_path / "gpu.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    assert idle_finetune.lock_is_available(lock_path)


# ---------------------------------------------------------------------------
# build_finetune_command
# ---------------------------------------------------------------------------

def test_build_finetune_command_default(monkeypatch):
    monkeypatch.delenv("KAIROS_IDLE_FINETUNE_CMD", raising=False)
    monkeypatch.delenv("KAIROS_IDLE_FINETUNE_MODEL", raising=False)
    monkeypatch.delenv("KAIROS_IDLE_FINETUNE_SYMBOL", raising=False)
    monkeypatch.delenv("KAIROS_IDLE_FINETUNE_OUTPUT", raising=False)

    cmd = idle_finetune.build_finetune_command()

    assert cmd[1:3] == ["run", "finetune"]
    assert "--model" in cmd and "kronos-small" in cmd
    assert "--symbol" in cmd and "BTC-USD" in cmd
    assert "--output-model" in cmd


def test_build_finetune_command_custom_model_symbol(monkeypatch):
    monkeypatch.delenv("KAIROS_IDLE_FINETUNE_CMD", raising=False)
    monkeypatch.setenv("KAIROS_IDLE_FINETUNE_MODEL", "kronos-base")
    monkeypatch.setenv("KAIROS_IDLE_FINETUNE_SYMBOL", "ETH-USD")
    monkeypatch.setenv("KAIROS_IDLE_FINETUNE_OUTPUT", "/tmp/my-model")

    cmd = idle_finetune.build_finetune_command()

    assert cmd[1] == "run"
    assert cmd[2] == "finetune"
    assert cmd[3:5] == ["--model", "kronos-base"]
    assert cmd[5:7] == ["--symbol", "ETH-USD"]
    assert cmd[7:9] == ["--output-model", "/tmp/my-model"]


def test_build_finetune_command_full_override(monkeypatch):
    monkeypatch.setenv(
        "KAIROS_IDLE_FINETUNE_CMD",
        "uv run ./finetune_csv/train_sequential.py --config configs/finetune_btc_base.yaml",
    )

    cmd = idle_finetune.build_finetune_command()

    assert cmd[1:3] == ["run", "./finetune_csv/train_sequential.py"]
    assert cmd[3:5] == ["--config", "configs/finetune_btc_base.yaml"]


# ---------------------------------------------------------------------------
# _env_flag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("YES", True), ("on", True)])
def test_env_flag_truthy(monkeypatch, value, expected):
    monkeypatch.setenv("KAIROS_TEST_FLAG", value)
    assert idle_finetune._env_flag("KAIROS_TEST_FLAG", False) is expected


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_env_flag_falsy(monkeypatch, value):
    monkeypatch.setenv("KAIROS_TEST_FLAG", value)
    assert idle_finetune._env_flag("KAIROS_TEST_FLAG", True) is False


def test_env_flag_default_when_unset(monkeypatch):
    monkeypatch.delenv("KAIROS_TEST_FLAG", raising=False)
    assert idle_finetune._env_flag("KAIROS_TEST_FLAG", True) is True
    assert idle_finetune._env_flag("KAIROS_TEST_FLAG", False) is False


# ---------------------------------------------------------------------------
# main() orchestration - fully mocked, no GPU/subprocess/network
# ---------------------------------------------------------------------------

def test_main_skips_silently_when_busy(monkeypatch, tmp_path):
    monkeypatch.setattr(idle_finetune, "is_gpu_idle", lambda **kw: False)
    monkeypatch.setattr(idle_finetune, "configure_logging", lambda: None)
    notify = mock.Mock()
    monkeypatch.setattr(idle_finetune, "_notify", notify)

    rc = idle_finetune.main([])

    assert rc == 0
    notify.assert_not_called()


def test_main_notifies_on_skip_when_requested(monkeypatch):
    monkeypatch.setattr(idle_finetune, "is_gpu_idle", lambda **kw: False)
    monkeypatch.setattr(idle_finetune, "configure_logging", lambda: None)
    notify = mock.Mock()
    monkeypatch.setattr(idle_finetune, "_notify", notify)

    rc = idle_finetune.main(["--notify-skip"])

    assert rc == 0
    notify.assert_called_once()


def test_main_skips_when_lock_held(monkeypatch):
    monkeypatch.setattr(idle_finetune, "is_gpu_idle", lambda **kw: True)
    monkeypatch.setattr(idle_finetune, "lock_is_available", lambda: False)
    monkeypatch.setattr(idle_finetune, "configure_logging", lambda: None)
    run_mock = mock.Mock()
    monkeypatch.setattr(idle_finetune.subprocess, "run", run_mock)

    rc = idle_finetune.main([])

    assert rc == 0
    run_mock.assert_not_called()


def test_main_runs_and_reports_success(monkeypatch):
    monkeypatch.setattr(idle_finetune, "is_gpu_idle", lambda **kw: True)
    monkeypatch.setattr(idle_finetune, "lock_is_available", lambda: True)
    monkeypatch.setattr(idle_finetune, "configure_logging", lambda: None)
    monkeypatch.setattr(idle_finetune, "require_gpu", lambda **kw: None)
    monkeypatch.setattr(idle_finetune, "build_finetune_command", lambda *a, **kw: ["echo", "hi"])

    class FakeLock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(idle_finetune, "GpuLock", lambda *a, **kw: FakeLock())

    completed = mock.Mock(returncode=0, stdout="ok", stderr="")
    run_mock = mock.Mock(return_value=completed)
    monkeypatch.setattr(idle_finetune.subprocess, "run", run_mock)

    notify = mock.Mock()
    monkeypatch.setattr(idle_finetune, "_notify", notify)

    rc = idle_finetune.main([])

    assert rc == 0
    run_mock.assert_called_once()
    assert notify.call_count == 2  # start + success


def test_main_returns_failure_exit_code_on_command_failure(monkeypatch):
    monkeypatch.setattr(idle_finetune, "is_gpu_idle", lambda **kw: True)
    monkeypatch.setattr(idle_finetune, "lock_is_available", lambda: True)
    monkeypatch.setattr(idle_finetune, "configure_logging", lambda: None)
    monkeypatch.setattr(idle_finetune, "require_gpu", lambda **kw: None)
    monkeypatch.setattr(idle_finetune, "build_finetune_command", lambda *a, **kw: ["false"])

    class FakeLock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(idle_finetune, "GpuLock", lambda *a, **kw: FakeLock())

    completed = mock.Mock(returncode=7, stdout="", stderr="boom")
    monkeypatch.setattr(idle_finetune.subprocess, "run", mock.Mock(return_value=completed))
    monkeypatch.setattr(idle_finetune, "_notify", mock.Mock())

    rc = idle_finetune.main([])

    assert rc == 7


def test_main_returns_1_on_gpu_or_lock_error(monkeypatch):
    monkeypatch.setattr(idle_finetune, "is_gpu_idle", lambda **kw: True)
    monkeypatch.setattr(idle_finetune, "lock_is_available", lambda: True)
    monkeypatch.setattr(idle_finetune, "configure_logging", lambda: None)

    def raise_ops_error(*a, **kw):
        raise idle_finetune.OpsError("GPU not healthy after recovery ladder")

    monkeypatch.setattr(idle_finetune, "require_gpu", raise_ops_error)

    class FakeLock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(idle_finetune, "GpuLock", lambda *a, **kw: FakeLock())
    monkeypatch.setattr(idle_finetune, "_notify", mock.Mock())

    rc = idle_finetune.main([])

    assert rc == 1


# ---------------------------------------------------------------------------
# get_finetune_symbols
# ---------------------------------------------------------------------------

def test_get_finetune_symbols_default_uses_price_cache_db(monkeypatch):
    monkeypatch.delenv("KAIROS_IDLE_FINETUNE_SYMBOLS", raising=False)
    monkeypatch.delenv("KAIROS_IDLE_FINETUNE_SYMBOL", raising=False)
    monkeypatch.setattr(
        idle_finetune, "_symbols_from_price_cache_db", lambda: ["AAPL", "TSLA", "BTC-USD"]
    )
    assert idle_finetune.get_finetune_symbols() == ["AAPL", "TSLA", "BTC-USD"]


def test_get_finetune_symbols_falls_back_to_default_when_db_empty(monkeypatch):
    monkeypatch.delenv("KAIROS_IDLE_FINETUNE_SYMBOLS", raising=False)
    monkeypatch.delenv("KAIROS_IDLE_FINETUNE_SYMBOL", raising=False)
    monkeypatch.setattr(idle_finetune, "_symbols_from_price_cache_db", lambda: [])
    assert idle_finetune.get_finetune_symbols() == idle_finetune.DEFAULT_FINETUNE_SYMBOLS


def test_get_finetune_symbols_single_env(monkeypatch):
    monkeypatch.delenv("KAIROS_IDLE_FINETUNE_SYMBOLS", raising=False)
    monkeypatch.setenv("KAIROS_IDLE_FINETUNE_SYMBOL", "ETH-USD")
    assert idle_finetune.get_finetune_symbols() == ["ETH-USD"]


def test_get_finetune_symbols_list_env(monkeypatch):
    monkeypatch.setenv("KAIROS_IDLE_FINETUNE_SYMBOLS", "BTC-USD, ETH-USD SOL-USD")
    assert idle_finetune.get_finetune_symbols() == ["BTC-USD", "ETH-USD", "SOL-USD"]


# ---------------------------------------------------------------------------
# Multi-symbol orchestration
# ---------------------------------------------------------------------------

def _make_main_mocks(monkeypatch, commands):
    """Patch idle_finetune.main dependencies; return subprocess run mock.

    ``commands`` is a list of (returncode, stderr) tuples returned sequentially.
    """
    monkeypatch.setattr(idle_finetune, "is_gpu_idle", lambda **kw: True)
    monkeypatch.setattr(idle_finetune, "lock_is_available", lambda: True)
    monkeypatch.setattr(idle_finetune, "configure_logging", lambda: None)
    monkeypatch.setattr(idle_finetune, "require_gpu", lambda **kw: None)

    class FakeLock:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(idle_finetune, "GpuLock", lambda *a, **kw: FakeLock())

    call_iter = iter(commands)

    def fake_run(*a, **kw):
        rc, stderr = next(call_iter)
        return mock.Mock(returncode=rc, stdout="ok", stderr=stderr)

    run_mock = mock.Mock(side_effect=fake_run)
    monkeypatch.setattr(idle_finetune.subprocess, "run", run_mock)
    return run_mock


def test_main_continues_on_data_error_and_tries_next_symbol(monkeypatch):
    monkeypatch.setattr(idle_finetune, "get_finetune_symbols", lambda: ["BTC-USD", "ETH-USD"])
    monkeypatch.setattr(idle_finetune, "_load_last_symbol", lambda symbols: None)
    monkeypatch.setattr(
        idle_finetune, "build_finetune_command", lambda symbol, *a, **kw: ["echo", symbol]
    )
    run_mock = _make_main_mocks(monkeypatch, [(1, "ERROR: no data returned for 'BTC-USD'"), (0, "")])

    notify = mock.Mock()
    monkeypatch.setattr(idle_finetune, "_notify", notify)

    rc = idle_finetune.main([])

    assert rc == 0
    assert run_mock.call_count == 2
    assert run_mock.call_args_list[0][0][0] == ["echo", "BTC-USD"]
    assert run_mock.call_args_list[1][0][0] == ["echo", "ETH-USD"]
    # start BTC + data-warning + start ETH + success
    assert notify.call_count == 4


def test_main_returns_zero_when_all_symbols_missing_data(monkeypatch):
    monkeypatch.setattr(idle_finetune, "get_finetune_symbols", lambda: ["BTC-USD", "ETH-USD"])
    monkeypatch.setattr(idle_finetune, "_load_last_symbol", lambda symbols: None)
    monkeypatch.setattr(
        idle_finetune, "build_finetune_command", lambda symbol, *a, **kw: ["echo", symbol]
    )
    run_mock = _make_main_mocks(
        monkeypatch,
        [(1, "ERROR: no data returned for 'BTC-USD'"), (1, "not enough data to build training")],
    )

    notify = mock.Mock()
    monkeypatch.setattr(idle_finetune, "_notify", notify)

    rc = idle_finetune.main([])

    assert rc == 0
    assert run_mock.call_count == 2
    # start + data-warning + start + data-warning + final summary
    assert notify.call_count == 5


def test_main_aborts_on_non_data_error(monkeypatch):
    monkeypatch.setattr(idle_finetune, "get_finetune_symbols", lambda: ["BTC-USD", "ETH-USD"])
    monkeypatch.setattr(idle_finetune, "_load_last_symbol", lambda symbols: None)
    monkeypatch.setattr(
        idle_finetune, "build_finetune_command", lambda symbol, *a, **kw: ["echo", symbol]
    )
    run_mock = _make_main_mocks(monkeypatch, [(1, "CUDA out of memory"), (0, "")])

    notify = mock.Mock()
    monkeypatch.setattr(idle_finetune, "_notify", notify)

    rc = idle_finetune.main([])

    assert rc == 1
    assert run_mock.call_count == 1
    # start + failure
    assert notify.call_count == 2


def test_rotate_symbols(monkeypatch):
    assert idle_finetune.rotate_symbols(["BTC-USD", "ETH-USD", "SOL-USD"], None) == [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
    ]
    assert idle_finetune.rotate_symbols(["BTC-USD", "ETH-USD", "SOL-USD"], "BTC-USD") == [
        "ETH-USD",
        "SOL-USD",
        "BTC-USD",
    ]
    assert idle_finetune.rotate_symbols(["BTC-USD", "ETH-USD", "SOL-USD"], "SOL-USD") == [
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
    ]


def test_main_rotates_after_success_and_saves_state(monkeypatch, tmp_path):
    state_file = tmp_path / "idle_finetune_state.json"
    monkeypatch.setattr(idle_finetune, "STATE_FILE", state_file)
    monkeypatch.setattr(idle_finetune, "get_finetune_symbols", lambda: ["BTC-USD", "ETH-USD"])
    monkeypatch.setattr(
        idle_finetune, "build_finetune_command", lambda symbol, *a, **kw: ["echo", symbol]
    )
    run_mock = _make_main_mocks(monkeypatch, [(0, "")])

    notify = mock.Mock()
    monkeypatch.setattr(idle_finetune, "_notify", notify)

    rc = idle_finetune.main([])
    assert rc == 0
    assert run_mock.call_count == 1
    assert run_mock.call_args_list[0][0][0] == ["echo", "BTC-USD"]
    state = json.loads(state_file.read_text())
    assert state["last_symbol"] == "BTC-USD"
    assert "BTC-USD" in state["last_trained"]

    # Next run should start with ETH-USD
    run_mock2 = _make_main_mocks(monkeypatch, [(0, "")])
    notify2 = mock.Mock()
    monkeypatch.setattr(idle_finetune, "_notify", notify2)
    rc = idle_finetune.main([])
    assert rc == 0
    assert run_mock2.call_count == 1
    assert run_mock2.call_args_list[0][0][0] == ["echo", "ETH-USD"]
    state = json.loads(state_file.read_text())
    assert state["last_symbol"] == "ETH-USD"
    assert "ETH-USD" in state["last_trained"]


def test_main_skips_symbol_on_cooldown(monkeypatch, tmp_path):
    state_file = tmp_path / "idle_finetune_state.json"
    monkeypatch.setattr(idle_finetune, "STATE_FILE", state_file)
    # Pretend BTC-USD was trained 1 hour ago; ETH-USD was never trained.
    trained_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    state_file.write_text(
        json.dumps({"last_symbol": "BTC-USD", "last_trained": {"BTC-USD": trained_at}})
    )
    monkeypatch.setattr(idle_finetune, "get_finetune_symbols", lambda: ["BTC-USD", "ETH-USD"])
    monkeypatch.setattr(
        idle_finetune, "build_finetune_command", lambda symbol, *a, **kw: ["echo", symbol]
    )
    run_mock = _make_main_mocks(monkeypatch, [(0, "")])

    notify = mock.Mock()
    monkeypatch.setattr(idle_finetune, "_notify", notify)

    # Default cooldown is 24h, so BTC is skipped and ETH is trained.
    rc = idle_finetune.main([])
    assert rc == 0
    assert run_mock.call_count == 1
    assert run_mock.call_args_list[0][0][0] == ["echo", "ETH-USD"]
    state = json.loads(state_file.read_text())
    assert state["last_symbol"] == "ETH-USD"
    assert "ETH-USD" in state["last_trained"]


def test_main_skips_all_symbols_when_on_cooldown(monkeypatch, tmp_path):
    state_file = tmp_path / "idle_finetune_state.json"
    monkeypatch.setattr(idle_finetune, "STATE_FILE", state_file)
    trained_at = datetime.now(timezone.utc).isoformat()
    state_file.write_text(
        json.dumps(
            {
                "last_symbol": "SOL-USD",
                "last_trained": {
                    "BTC-USD": trained_at,
                    "ETH-USD": trained_at,
                    "SOL-USD": trained_at,
                },
            }
        )
    )
    monkeypatch.setattr(idle_finetune, "get_finetune_symbols", lambda: ["BTC-USD", "ETH-USD", "SOL-USD"])
    monkeypatch.setattr(idle_finetune, "is_gpu_idle", lambda **kw: True)
    monkeypatch.setattr(idle_finetune, "lock_is_available", lambda: True)
    monkeypatch.setattr(idle_finetune, "configure_logging", lambda: None)
    run_mock = mock.Mock()
    monkeypatch.setattr(idle_finetune.subprocess, "run", run_mock)

    notify = mock.Mock()
    monkeypatch.setattr(idle_finetune, "_notify", notify)

    rc = idle_finetune.main(["--notify-skip"])
    assert rc == 0
    run_mock.assert_not_called()
    notify.assert_called_once()
    assert "recently trained" in notify.call_args[0][0].lower()
