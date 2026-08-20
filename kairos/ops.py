"""Operational utilities for Kairos automation.

Shared helpers used by scheduled runners and the live-ops layer:

* GPU advisory lock so only one GPU-bound Kairos job runs at a time.
* GPU health probe and optional recovery-ladder wrapper.
* Telegram notifications for signals, failures, and completions.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import requests

from kairos.errors import KairosError


class OpsError(KairosError):
    """Operational/automation error."""


DEFAULT_LOCK_PATH = Path("/tmp/kairos_gpu.lock")
DEFAULT_LOCK_TIMEOUT_SECONDS = 300  # wait up to 5 minutes for the GPU
DEFAULT_GPU_UTIL_THRESHOLD = 10
DEFAULT_IDLE_SAMPLES = 3
DEFAULT_IDLE_SAMPLE_INTERVAL_SECONDS = 10.0  # 3 samples * 10s = ~30s idle window
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_DOCUMENT_API_URL = "https://api.telegram.org/bot{token}/sendDocument"


class GpuLock:
    """Process-wide advisory lock for GPU-bound Kairos jobs.

    Use as a context manager. Multiple Kairos runners (signals, discovery,
    paper trading, finetuning) share the same lock path so the RTX 3060 never
    runs two GPU workloads simultaneously.
    """

    def __init__(
        self,
        lock_path: Path | str = DEFAULT_LOCK_PATH,
        timeout: int = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ):
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self._fd: Optional[int] = None

    def __enter__(self) -> "GpuLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        start = time.time()
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - start > self.timeout:
                    raise OpsError(
                        f"Could not acquire GPU lock {self.lock_path} within "
                        f"{self.timeout}s (another Kairos GPU job is running)"
                    )
                time.sleep(min(0.5, self.timeout / 2.0 or 0.5))
        os.ftruncate(self._fd, 0)
        os.write(self._fd, f"{os.getpid()}\n".encode())
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None
        return False


def gpu_healthy() -> bool:
    """Return True if nvidia-smi works and torch sees CUDA in a fresh process."""
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return False
    result = subprocess.run(
        [sys.executable, "-c", "import torch; assert torch.cuda.is_available()"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result.returncode == 0


def recover_gpu(allow_reboot: bool = False) -> bool:
    """Run scripts/gpu_recover.py ladder and return whether GPU is healthy.

    Exit 0 from gpu_recover.py means the GPU is healthy (possibly after
    recovery). Exit 3 means a reboot was scheduled.
    """
    cmd = [sys.executable, "scripts/gpu_recover.py"]
    if allow_reboot:
        cmd.append("--allow-reboot")
    result = subprocess.run(cmd, timeout=300, capture_output=True, text=True)
    return result.returncode == 0


def require_gpu(
    allow_recover: bool = True,
    allow_reboot: bool = False,
) -> None:
    """Raise OpsError if GPU is not available after optional recovery."""
    if gpu_healthy():
        return
    if not allow_recover:
        raise OpsError("GPU not healthy and recovery is disabled")
    if recover_gpu(allow_reboot=allow_reboot):
        return
    raise OpsError("GPU not healthy after recovery ladder")


def gpu_utilization_percent() -> Optional[int]:
    """Return current GPU utilization percent from nvidia-smi, or None."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


def is_gpu_idle(
    threshold: int = DEFAULT_GPU_UTIL_THRESHOLD,
    samples: int = DEFAULT_IDLE_SAMPLES,
    interval_seconds: float = DEFAULT_IDLE_SAMPLE_INTERVAL_SECONDS,
    util_fn: Callable[[], Optional[int]] = gpu_utilization_percent,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bool:
    """Return True iff GPU utilization stays below `threshold` for `samples`
    consecutive reads, `interval_seconds` apart.

    A missing/unreadable utilization sample (nvidia-smi failure) is treated as
    "not idle" -- callers should not start new GPU work when they can't
    confirm the GPU is actually free. Returns as soon as one sample is over
    threshold, without waiting out the remaining samples.

    Used by kairos_pipeline.py's --stage finetune_next so it doesn't start
    training on a busy GPU.
    """
    for i in range(samples):
        util = util_fn()
        if util is None or util >= threshold:
            return False
        if i < samples - 1:
            sleep_fn(interval_seconds)
    return True


def send_telegram(
    text: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
    parse_mode: Optional[str] = "Markdown",
) -> None:
    """Send a plain-text Telegram message using the Bot API.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from the environment unless
    overridden. Raises OpsError if credentials are missing or the API call
    fails.

    Pass `parse_mode=None` to send as plain text (the `parse_mode` field is
    omitted from the request entirely, not sent as a JSON null): callers
    embedding dynamic/uncontrolled content -- asset symbols, stderr tails,
    tracebacks -- should do this, since a single unbalanced Markdown special
    character anywhere in that content makes Telegram reject the whole
    message with a 400 "can't parse entities" error.
    """
    print(text)
    bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        raise OpsError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set to send Telegram messages"
        )

    url = TELEGRAM_API_URL.format(token=bot_token)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        raise OpsError(f"Telegram API HTTP error {exc.code}: {exc.read().decode()}") from exc
    except Exception as exc:
        raise OpsError(f"Telegram API call failed: {exc}") from exc


def send_telegram_document(
    file_path: Path | str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None,
) -> None:
    """Send a file as a Telegram document attachment using the Bot API.

    Reads TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID from the environment unless
    overridden. Raises OpsError if credentials are missing, the file doesn't
    exist, or the API call fails. No caption/parse_mode is sent -- keeps
    this free of the unbalanced-Markdown risk send_telegram() warns about.
    """
    bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        raise OpsError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set to send Telegram messages"
        )

    file_path = Path(file_path)
    if not file_path.is_file():
        raise OpsError(f"Telegram document attachment not found: {file_path}")

    url = TELEGRAM_DOCUMENT_API_URL.format(token=bot_token)
    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                url,
                data={"chat_id": chat_id},
                files={"document": (file_path.name, f)},
                timeout=30,
            )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise OpsError(f"Telegram document upload failed: {exc}") from exc
