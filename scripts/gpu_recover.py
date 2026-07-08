#!/usr/bin/env python3
"""GPU/CUDA recovery tool with an escalation ladder (L0-L4).

Standalone, stdlib-only. Diagnoses whether CUDA is usable and, if not,
escalates through progressively more disruptive recovery actions:

    L0  diagnose             (nvidia-smi + fresh-process torch probe)
    L1  free the GPU         (kill non-display compute processes)
    L2  UVM reload           (rmmod/modprobe nvidia_uvm, non-disruptive)
    L3  full module reload   (bounce display-manager + all nvidia modules)
    L4  reboot + resume      (only with --allow-reboot / KAIROS_GPU_ALLOW_REBOOT=1)

Exit codes:
    0   GPU healthy (possibly after recovery)
    2   unrecoverable and reboot not permitted
    3   reboot scheduled (informational; process ends at reboot)

All side effects (kill/rmmod/modprobe/systemctl/reboot) go through the
injectable `Runner` so tests and --dry-run never touch the real system.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

STATE_DIR = Path.home() / ".local" / "state" / "kairos"
LOG_PATH = STATE_DIR / "gpu_recover.log"
RESUME_SCRIPT_PATH = STATE_DIR / "resume.sh"
RESUME_LOG_PATH = STATE_DIR / "resume.log"
SYSTEMD_USER_DIR = Path.home() / ".config" / "systemd" / "user"
RESUME_UNIT_PATH = SYSTEMD_USER_DIR / "kairos-resume.service"
RESUME_UNIT_NAME = "kairos-resume.service"

EXIT_HEALTHY = 0
EXIT_UNRECOVERABLE = 2
EXIT_REBOOT_SCHEDULED = 3

DISPLAY_PROCESS_NAMES = {"xorg", "x", "gdm", "gdm3", "sddm", "lightdm", "display-manager"}


@dataclass
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Runner:
    """Real command runner. All side effects live here."""

    def run(self, cmd: Sequence[str], timeout: Optional[float] = None) -> CommandResult:
        try:
            proc = subprocess.run(
                list(cmd), capture_output=True, text=True, timeout=timeout
            )
            return CommandResult(proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(124, exc.stdout or "", exc.stderr or "timeout")
        except FileNotFoundError as exc:
            return CommandResult(127, "", str(exc))

    def kill(self, pid: int, sig: int) -> None:
        try:
            os.kill(pid, sig)
        except OSError:
            pass

    def write_file(self, path: Path, content: str, mode: int = 0o644) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        os.chmod(path, mode)


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr)
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ── L0: diagnose ────────────────────────────────────────────────────────────

def check_nvidia_smi(runner: Runner) -> bool:
    result = runner.run(["nvidia-smi"], timeout=10)
    return result.ok


def check_torch_cuda_fresh(runner: Runner) -> bool:
    """Probe CUDA availability in a fresh child process (torch caches CUDA init state)."""
    result = runner.run(
        ["uv", "run", "python", "-c", "import torch; assert torch.cuda.is_available()"],
        timeout=60,
    )
    return result.ok


def gpu_healthy(runner: Runner) -> bool:
    return check_nvidia_smi(runner) and check_torch_cuda_fresh(runner)


# ── L1: free the GPU ────────────────────────────────────────────────────────

def list_compute_processes(runner: Runner) -> List[dict]:
    result = runner.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        timeout=10,
    )
    procs = []
    if not result.ok:
        return procs
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        pid_str, name = parts[0], ",".join(parts[1:])
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        procs.append({"pid": pid, "name": name})
    return procs


def is_excluded(proc: dict, exclude_pids: set) -> bool:
    if proc["pid"] in exclude_pids:
        return True
    name_lower = proc["name"].lower()
    return any(d in name_lower for d in DISPLAY_PROCESS_NAMES)


def l1_free_gpu(runner: Runner, dry_run: bool, exclude_pids: set) -> None:
    procs = list_compute_processes(runner)
    targets = [p for p in procs if not is_excluded(p, exclude_pids)]
    if not targets:
        log("L1: no killable compute processes found")
        return
    for p in targets:
        log(f"L1: SIGTERM pid={p['pid']} name={p['name']}")
        if not dry_run:
            runner.kill(p["pid"], 15)  # SIGTERM
    if not dry_run:
        time.sleep(2)
    still_alive = list_compute_processes(runner)
    for p in still_alive:
        if is_excluded(p, exclude_pids):
            continue
        log(f"L1: SIGKILL pid={p['pid']} name={p['name']}")
        if not dry_run:
            runner.kill(p["pid"], 9)  # SIGKILL


# ── L2: UVM reload (non-disruptive) ─────────────────────────────────────────

def l2_uvm_reload(runner: Runner, dry_run: bool) -> None:
    log("L2: sudo rmmod nvidia_uvm")
    if not dry_run:
        runner.run(["sudo", "rmmod", "nvidia_uvm"], timeout=30)
    log("L2: sudo modprobe nvidia_uvm")
    if not dry_run:
        runner.run(["sudo", "modprobe", "nvidia_uvm"], timeout=30)


# ── L3: full module reload (disruptive) ─────────────────────────────────────

def l3_full_reload(runner: Runner, dry_run: bool) -> None:
    log("L3: WARNING - this bounces display-manager and kills the X session")
    log("L3: sudo systemctl stop display-manager")
    if not dry_run:
        runner.run(["sudo", "systemctl", "stop", "display-manager"], timeout=30)
    modules = ["nvidia_uvm", "nvidia_drm", "nvidia_modeset", "nvidia"]
    for m in modules:
        log(f"L3: sudo rmmod {m}")
        if not dry_run:
            runner.run(["sudo", "rmmod", m], timeout=30)
    for m in reversed(modules):
        log(f"L3: sudo modprobe {m}")
        if not dry_run:
            runner.run(["sudo", "modprobe", m], timeout=30)
    log("L3: sudo systemctl start display-manager")
    if not dry_run:
        runner.run(["sudo", "systemctl", "start", "display-manager"], timeout=30)


# ── L4: reboot + resume ──────────────────────────────────────────────────────

RESUME_SH_TEMPLATE = """#!/bin/bash
cd {resume_cwd} && exec {resume_cmd} >> {resume_log} 2>&1
"""

RESUME_UNIT_TEMPLATE = """[Unit]
Description=Kairos GPU-recovery resume

[Service]
Type=oneshot
ExecStart=/bin/bash {resume_script}
ExecStartPost=/usr/bin/systemctl --user disable {unit_name}

[Install]
WantedBy=default.target
"""


def render_resume_script(resume_cmd: str, resume_cwd: str) -> str:
    return RESUME_SH_TEMPLATE.format(
        resume_cwd=resume_cwd, resume_cmd=resume_cmd, resume_log=str(RESUME_LOG_PATH)
    )


def render_resume_unit(resume_script_path: Path) -> str:
    return RESUME_UNIT_TEMPLATE.format(
        resume_script=str(resume_script_path), unit_name=RESUME_UNIT_NAME
    )


def write_resume_files(runner: Runner, resume_cmd: str, resume_cwd: str) -> None:
    script_content = render_resume_script(resume_cmd, resume_cwd)
    unit_content = render_resume_unit(RESUME_SCRIPT_PATH)
    log(f"L4: writing resume script to {RESUME_SCRIPT_PATH}")
    runner.write_file(RESUME_SCRIPT_PATH, script_content, mode=0o755)
    log(f"L4: writing resume unit to {RESUME_UNIT_PATH}")
    runner.write_file(RESUME_UNIT_PATH, unit_content, mode=0o644)


def l4_reboot_and_resume(
    runner: Runner, dry_run: bool, resume_cmd: Optional[str], resume_cwd: str
) -> None:
    if resume_cmd:
        if dry_run:
            script_content = render_resume_script(resume_cmd, resume_cwd)
            unit_content = render_resume_unit(RESUME_SCRIPT_PATH)
            log(f"L4: (dry-run) would write resume script to {RESUME_SCRIPT_PATH}:\n{script_content}")
            log(f"L4: (dry-run) would write resume unit to {RESUME_UNIT_PATH}:\n{unit_content}")
        else:
            write_resume_files(runner, resume_cmd, resume_cwd)
        log(f"L4: sudo systemctl --user enable {RESUME_UNIT_NAME}")
        if not dry_run:
            runner.run(["systemctl", "--user", "enable", RESUME_UNIT_NAME], timeout=10)
    log("L4: sudo systemctl reboot")
    if not dry_run:
        runner.run(["sudo", "systemctl", "reboot"], timeout=10)


# ── Ladder orchestration ─────────────────────────────────────────────────────

def run_ladder(
    runner: Runner,
    max_level: int = 4,
    dry_run: bool = False,
    check_only: bool = False,
    allow_reboot: bool = False,
    resume_cmd: Optional[str] = None,
    resume_cwd: Optional[str] = None,
    exclude_pids: Optional[set] = None,
) -> int:
    exclude_pids = exclude_pids or set()
    exclude_pids.add(os.getpid())

    log("L0: diagnose")
    if gpu_healthy(runner):
        log("L0: GPU healthy")
        return EXIT_HEALTHY
    log("L0: GPU unhealthy")

    if check_only:
        log("check-only: not attempting recovery")
        return EXIT_UNRECOVERABLE

    if max_level >= 1:
        log("L1: free the GPU")
        l1_free_gpu(runner, dry_run, exclude_pids)
        if not dry_run and gpu_healthy(runner):
            log("L1: GPU healthy after freeing processes")
            return EXIT_HEALTHY

    if max_level >= 2:
        log("L2: UVM reload")
        l2_uvm_reload(runner, dry_run)
        if not dry_run and gpu_healthy(runner):
            log("L2: GPU healthy after UVM reload")
            return EXIT_HEALTHY

    if max_level >= 3:
        log("L3: full module reload")
        l3_full_reload(runner, dry_run)
        if not dry_run and gpu_healthy(runner):
            log("L3: GPU healthy after full module reload")
            return EXIT_HEALTHY

    if max_level >= 4:
        if not allow_reboot:
            log("L4: reboot required but --allow-reboot not set; stopping")
            return EXIT_UNRECOVERABLE
        log("L4: reboot + resume")
        l4_reboot_and_resume(
            runner, dry_run, resume_cmd, resume_cwd or str(Path.cwd())
        )
        return EXIT_REBOOT_SCHEDULED

    log("Ladder exhausted at configured --max-level without recovery")
    return EXIT_UNRECOVERABLE


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                         help="Only run L0 diagnostics; do not attempt recovery")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print every action at every level without executing side effects")
    parser.add_argument("--allow-reboot", action="store_true",
                         help="Permit L4 reboot (also settable via KAIROS_GPU_ALLOW_REBOOT=1)")
    parser.add_argument("--resume-cmd", default=None,
                         help="Command to re-run after reboot via a systemd user resume unit")
    parser.add_argument("--resume-cwd", default=None,
                         help="Working directory for --resume-cmd (default: cwd)")
    parser.add_argument("--max-level", type=int, default=4,
                         help="Highest ladder level to attempt (0-4, default 4)")
    return parser


def main(argv: Optional[Sequence[str]] = None, runner: Optional[Runner] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runner = runner or Runner()

    allow_reboot = args.allow_reboot or os.environ.get("KAIROS_GPU_ALLOW_REBOOT") == "1"

    code = run_ladder(
        runner,
        max_level=args.max_level,
        dry_run=args.dry_run,
        check_only=args.check_only,
        allow_reboot=allow_reboot,
        resume_cmd=args.resume_cmd,
        resume_cwd=args.resume_cwd,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
