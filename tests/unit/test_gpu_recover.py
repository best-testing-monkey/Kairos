"""Tests for scripts/gpu_recover.py and strategy/kairos_gpu.py.

All tests use an injected fake Runner - no real nvidia-smi/sudo/systemctl/
kill calls ever happen. File-rendering tests use a tmp HOME.
"""
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "gpu_recover.py"

spec = importlib.util.spec_from_file_location("gpu_recover", SCRIPT_PATH)
gpu_recover = importlib.util.module_from_spec(spec)
sys.modules["gpu_recover"] = gpu_recover
spec.loader.exec_module(gpu_recover)


class FakeRunner:
    """Records every call instead of executing it. Never touches the real system."""

    def __init__(self, responses=None, compute_procs=None):
        # responses: dict mapping a command-key (tuple of first two argv tokens,
        # or a predicate) -> CommandResult-like namedtuple substitute
        self.calls = []
        self.kills = []
        self.writes = {}
        self._responses = responses or {}
        self._compute_procs_calls = 0
        self._compute_procs = compute_procs if compute_procs is not None else []

    def _key(self, cmd):
        return tuple(cmd)

    def run(self, cmd, timeout=None):
        self.calls.append(list(cmd))
        cmd = list(cmd)
        if cmd[:1] == ["nvidia-smi"] and "--query-compute-apps=pid,process_name" in cmd:
            csv_lines = "\n".join(f"{p['pid']}, {p['name']}" for p in self._compute_procs)
            return gpu_recover.CommandResult(0, csv_lines, "")
        key = self._key(cmd)
        if key in self._responses:
            return self._responses[key]
        # default: match by prefix
        for k, v in self._responses.items():
            if tuple(cmd[: len(k)]) == k:
                return v
        return gpu_recover.CommandResult(0, "", "")

    def kill(self, pid, sig):
        self.kills.append((pid, sig))

    def write_file(self, path, content, mode=0o644):
        self.writes[str(path)] = content


def healthy_responses():
    return {
        ("nvidia-smi",): gpu_recover.CommandResult(0, "GPU 0: healthy", ""),
        ("uv", "run", "python", "-c"): gpu_recover.CommandResult(0, "", ""),
    }


def unhealthy_responses():
    return {
        ("nvidia-smi",): gpu_recover.CommandResult(0, "GPU 0: present", ""),
        ("uv", "run", "python", "-c"): gpu_recover.CommandResult(1, "", "no cuda"),
    }


class TestLadderHealthyShortCircuit:
    def test_healthy_at_l0_short_circuits(self):
        runner = FakeRunner(responses=healthy_responses())
        code = gpu_recover.run_ladder(runner)
        assert code == gpu_recover.EXIT_HEALTHY
        # Only L0 diagnostic calls should have happened - no kill/rmmod/reboot.
        assert runner.kills == []
        joined = [" ".join(c) for c in runner.calls]
        assert not any("rmmod" in c or "reboot" in c for c in joined)


class TestL1KillList:
    def test_l1_excludes_xorg_and_self(self):
        procs = [
            {"pid": 111, "name": "Xorg"},
            {"pid": 222, "name": "python_training_job"},
            {"pid": os.getpid(), "name": "gpu_recover_self"},
        ]
        responses = unhealthy_responses()
        runner = FakeRunner(responses=responses, compute_procs=procs)
        gpu_recover.run_ladder(runner, max_level=1)
        killed_pids = {pid for pid, _ in runner.kills}
        assert 222 in killed_pids
        assert 111 not in killed_pids
        assert os.getpid() not in killed_pids


class TestLadderOrder:
    def test_order_l1_l2_l3_l4_when_all_unhealthy(self):
        responses = unhealthy_responses()
        runner = FakeRunner(responses=responses, compute_procs=[])
        code = gpu_recover.run_ladder(
            runner, max_level=4, allow_reboot=True, resume_cmd="echo hi", resume_cwd="/tmp"
        )
        joined = [" ".join(c) for c in runner.calls]
        # L1 handled via kill (no subprocess call needed since no procs), but
        # L2/L3 rmmod and L4 reboot should appear in that order.
        rmmod_calls = [c for c in joined if "rmmod" in c]
        reboot_calls = [c for c in joined if "reboot" in c]
        assert any("nvidia_uvm" in c for c in rmmod_calls)
        assert len(reboot_calls) >= 1
        assert code == gpu_recover.EXIT_REBOOT_SCHEDULED


class TestNoRebootPermission:
    def test_exits_2_without_reboot_when_not_permitted(self):
        responses = unhealthy_responses()
        runner = FakeRunner(responses=responses, compute_procs=[])
        code = gpu_recover.run_ladder(runner, max_level=4, allow_reboot=False)
        assert code == gpu_recover.EXIT_UNRECOVERABLE
        joined = [" ".join(c) for c in runner.calls]
        assert not any("reboot" in c for c in joined)


class TestDryRun:
    def test_dry_run_performs_zero_side_effect_calls(self):
        procs = [{"pid": 222, "name": "python_job"}]
        responses = unhealthy_responses()
        runner = FakeRunner(responses=responses, compute_procs=procs)
        code = gpu_recover.run_ladder(
            runner, max_level=4, dry_run=True, allow_reboot=True,
            resume_cmd="echo hi", resume_cwd="/tmp",
        )
        # No kills should have happened.
        assert runner.kills == []
        # No rmmod/modprobe/systemctl/reboot calls should have happened.
        joined = [" ".join(c) for c in runner.calls]
        assert not any(
            any(tok in c for tok in ("rmmod", "modprobe", "systemctl", "reboot"))
            for c in joined
        )
        # No files written under dry-run.
        assert runner.writes == {}
        assert code == gpu_recover.EXIT_REBOOT_SCHEDULED


class TestCheckOnly:
    def test_check_only_stops_after_l0(self):
        responses = unhealthy_responses()
        runner = FakeRunner(responses=responses, compute_procs=[{"pid": 222, "name": "x"}])
        code = gpu_recover.run_ladder(runner, check_only=True)
        assert code == gpu_recover.EXIT_UNRECOVERABLE
        assert runner.kills == []


class TestResumeFileRendering:
    def test_resume_script_and_unit_rendered_correctly(self, tmp_path, monkeypatch):
        # Point the module's path constants at a tmp HOME instead of reloading
        # (the module was loaded via spec_from_file_location, not a normal
        # import, so importlib.reload can't re-resolve its spec).
        tmp_state_dir = tmp_path / "state" / "kairos"
        tmp_systemd_dir = tmp_path / "config" / "systemd" / "user"
        monkeypatch.setattr(gpu_recover, "RESUME_SCRIPT_PATH", tmp_state_dir / "resume.sh")
        monkeypatch.setattr(gpu_recover, "RESUME_LOG_PATH", tmp_state_dir / "resume.log")
        monkeypatch.setattr(gpu_recover, "RESUME_UNIT_PATH", tmp_systemd_dir / "kairos-resume.service")

        runner = FakeRunner()
        gpu_recover.write_resume_files(runner, "echo resumed", "/some/cwd")
        assert str(gpu_recover.RESUME_SCRIPT_PATH) in runner.writes
        assert str(gpu_recover.RESUME_UNIT_PATH) in runner.writes
        script_content = runner.writes[str(gpu_recover.RESUME_SCRIPT_PATH)]
        unit_content = runner.writes[str(gpu_recover.RESUME_UNIT_PATH)]
        assert "cd /some/cwd" in script_content
        assert "echo resumed" in script_content
        assert "Type=oneshot" in unit_content
        assert "kairos-resume.service" in unit_content
        assert "WantedBy=default.target" in unit_content


class TestCLIMainDryRun:
    def test_main_dry_run_allow_reboot_no_side_effects(self):
        responses = unhealthy_responses()
        runner = FakeRunner(responses=responses, compute_procs=[])
        code = gpu_recover.main(
            ["--dry-run", "--allow-reboot", "--resume-cmd", "echo resumed"],
            runner=runner,
        )
        assert code == gpu_recover.EXIT_REBOOT_SCHEDULED
        assert runner.kills == []
        assert runner.writes == {}


# ── strategy/kairos_gpu.py ───────────────────────────────────────────────────

import kairos_gpu


class TestEnsureCuda:
    def test_allow_cpu_short_circuits_without_invoking_recovery(self, monkeypatch):
        monkeypatch.setenv("KAIROS_ALLOW_CPU", "1")
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("subprocess.run") as mock_run:
                result = kairos_gpu.ensure_cuda()
        assert result is False
        mock_run.assert_not_called()
        monkeypatch.delenv("KAIROS_ALLOW_CPU", raising=False)

    def test_cuda_already_available_returns_true_without_subprocess(self, monkeypatch):
        monkeypatch.delenv("KAIROS_ALLOW_CPU", raising=False)
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("subprocess.run") as mock_run:
                result = kairos_gpu.ensure_cuda()
        assert result is True
        mock_run.assert_not_called()

    def test_recovery_invoked_with_correct_resume_cmd(self, monkeypatch):
        monkeypatch.delenv("KAIROS_ALLOW_CPU", raising=False)
        monkeypatch.delenv("KAIROS_GPU_ALLOW_REBOOT", raising=False)
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_result = MagicMock()
        fake_result.returncode = 0
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("subprocess.run", return_value=fake_result) as mock_run:
                with pytest.raises(SystemExit) as exc_info:
                    kairos_gpu.ensure_cuda(resume_cmd=["my_script.py", "--foo"])
        assert exc_info.value.code == 75
        called_cmd = mock_run.call_args[0][0]
        assert called_cmd[:3] == ["uv", "run", kairos_gpu.GPU_RECOVER_SCRIPT]
        assert "--resume-cmd" in called_cmd
        idx = called_cmd.index("--resume-cmd")
        assert called_cmd[idx + 1] == "my_script.py --foo"

    def test_allow_reboot_env_passed_through(self, monkeypatch):
        monkeypatch.delenv("KAIROS_ALLOW_CPU", raising=False)
        monkeypatch.setenv("KAIROS_GPU_ALLOW_REBOOT", "1")
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_result = MagicMock()
        fake_result.returncode = 0
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("subprocess.run", return_value=fake_result) as mock_run:
                with pytest.raises(SystemExit):
                    kairos_gpu.ensure_cuda(resume_cmd=["x.py"])
        called_cmd = mock_run.call_args[0][0]
        assert "--allow-reboot" in called_cmd
        monkeypatch.delenv("KAIROS_GPU_ALLOW_REBOOT", raising=False)

    def test_recovery_failure_raises_runtime_error(self, monkeypatch):
        monkeypatch.delenv("KAIROS_ALLOW_CPU", raising=False)
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_result = MagicMock()
        fake_result.returncode = 2
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("subprocess.run", return_value=fake_result):
                with pytest.raises(RuntimeError):
                    kairos_gpu.ensure_cuda(resume_cmd=["x.py"])


# ── kairos_pipeline retry-on-75 ──────────────────────────────────────────────

import kairos_pipeline


class TestPipelineRetryOn75:
    def test_exit_75_retries_once_then_succeeds(self, tmp_path, monkeypatch):
        export_payload = {"summary": {}, "strategy_rankings": [], "shadow_performance": {}}

        calls = {"n": 0}

        def fake_run(cmd, cwd=None, capture_output=None, text=None, env=None):
            calls["n"] += 1
            # kairos_pipeline writes to the tmp_path passed via --export_json;
            # find it in cmd and populate it so json.load succeeds.
            idx = cmd.index("--export_json")
            out_path = cmd[idx + 1]
            import json
            with open(out_path, "w") as f:
                json.dump(export_payload, f)
            result = MagicMock()
            result.returncode = 75 if calls["n"] == 1 else 0
            result.stdout = ""
            result.stderr = ""
            return result

        with patch("kairos_pipeline.subprocess.run", side_effect=fake_run):
            payload = kairos_pipeline.run_backtest_subprocess(["BTC-USD"], no_prediction=True)

        assert calls["n"] == 2
        assert payload == export_payload
