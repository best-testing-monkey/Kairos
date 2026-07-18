"""GPU/CUDA opt-out strict-mode helper.

CUDA is required by default. Callers wanting the legacy silent CPU
fallback must set KAIROS_ALLOW_CPU=1. Otherwise, when CUDA is missing,
this module shells out to scripts/gpu_recover.py's escalation ladder
(never imports it directly - the recovery script must stay a standalone
subprocess so its side effects are isolated from the caller's process).

See docs/plan and CLAUDE.md "GPU recovery" gotcha for exit-code semantics.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
GPU_RECOVER_SCRIPT = str(REPO_ROOT / "scripts" / "gpu_recover.py")

EX_TEMPFAIL = 75


def cuda_available_fresh() -> bool:
    """Probe CUDA availability in a fresh subprocess (torch caches CUDA init state)."""
    result = subprocess.run(
        ["uv", "run", "python", "-c", "import torch; assert torch.cuda.is_available()"],
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0


def ensure_cuda(resume_cmd: Optional[List[str]] = None) -> bool:
    """Ensure CUDA is usable, invoking the recovery ladder if needed.

    Returns True if the caller should proceed on GPU. Returns False only
    when KAIROS_ALLOW_CPU=1 is set (legacy CPU-fallback behavior). If
    recovery heals the GPU but the *current* process still can't see it
    (torch caches CUDA init state), exits with code 75 (EX_TEMPFAIL) so a
    wrapper/pipeline can retry in a fresh process.
    """
    import torch

    if torch.cuda.is_available():
        return True

    if os.environ.get("KAIROS_ALLOW_CPU") == "1":
        return False

    cmd = ["uv", "run", GPU_RECOVER_SCRIPT]
    resume = resume_cmd if resume_cmd is not None else list(sys.argv)
    if resume:
        cmd.extend(["--resume-cmd", " ".join(resume), "--resume-cwd", os.getcwd()])
    if os.environ.get("KAIROS_GPU_ALLOW_REBOOT") == "1":
        cmd.append("--allow-reboot")

    print(f"CUDA unavailable and KAIROS_ALLOW_CPU is not set; invoking recovery: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(
            f"GPU recovery failed (exit {result.returncode}) and KAIROS_ALLOW_CPU is not set. "
            "Set KAIROS_ALLOW_CPU=1 to fall back to CPU, or resolve the GPU issue."
        )

    # Recovery reported success, but this process cached the old CUDA state.
    # A fresh process will see the healed GPU - ask the caller to retry.
    print(
        "GPU recovered by scripts/gpu_recover.py, but this process's torch state is stale. "
        "Exiting with code 75 (EX_TEMPFAIL) so the caller can retry in a fresh process."
    )
    sys.exit(EX_TEMPFAIL)
