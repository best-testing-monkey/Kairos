#!/usr/bin/env python3
"""Idle-GPU fine-tuning runner with Telegram alerts.

Intended to be triggered by a systemd user timer every ~30 minutes. It:

1. Samples GPU utilization several times in a row; if it stays below a
   threshold for all of them, the RTX 3060 is considered idle.
2. Probes whether the shared GPU lock (``kairos.ops.GpuLock``) is free.
3. If both hold, acquires the lock, verifies CUDA health (with the usual
   recovery ladder), and runs one fine-tuning command for the next symbol that
   is not currently within its cooldown window.
4. Rotates through ``DEFAULT_FINETUNE_SYMBOLS`` and records the last successful
   symbol and training time, so the same instrument is not retrained until
   ``KAIROS_IDLE_FINETUNE_COOLDOWN_SECONDS`` (default 24h) has elapsed.
5. Sends Telegram alerts on start, success, and failure. A "too busy, skipped
   this cycle" outcome is silent unless ``--notify-skip``/
   ``KAIROS_IDLE_NOTIFY_SKIP`` is set.

Utilization/lock checks intentionally run *before* the CUDA health check: a
busy-but-healthy GPU should never trigger the recovery ladder (which can kill
GPU processes), so ``require_gpu()`` only runs once the lock is actually held
by this process, mirroring ``kairos_daily_signals.py``/``kairos_weekly_discovery.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Repo root is two levels up from this script.
REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

from kairos.ops import (  # noqa: E402
    DEFAULT_GPU_UTIL_THRESHOLD,
    DEFAULT_LOCK_PATH,
    GpuLock,
    OpsError,
    gpu_utilization_percent,
    is_gpu_idle,
    require_gpu,
    send_telegram,
)


STATE_DIR = Path.home() / ".local" / "state" / "kairos"
LOG_PATH = STATE_DIR / "idle_finetune.log"
STATE_FILE = STATE_DIR / "idle_finetune_state.json"

DEFAULT_SAMPLES = 3
DEFAULT_SAMPLE_INTERVAL_SECONDS = 10.0  # 3 samples * 10s = ~30s idle window

DEFAULT_FINETUNE_MODEL = "kronos-small"
DEFAULT_FINETUNE_SYMBOL = "BTC-USD"
DEFAULT_FINETUNE_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]
DEFAULT_FINETUNE_COOLDOWN_SECONDS = 24 * 60 * 60  # 24 hours


logger = logging.getLogger("kairos_idle_finetune")


def configure_logging() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH),
            logging.StreamHandler(sys.stdout),
        ],
    )


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def lock_is_available(lock_path: Path | str = DEFAULT_LOCK_PATH) -> bool:
    """Non-blocking probe: True iff the shared GPU lock is currently free."""
    try:
        with GpuLock(lock_path=lock_path, timeout=0):
            pass
    except OpsError:
        return False
    return True


def get_finetune_symbols() -> list[str]:
    """Return the list of symbols to fine-tune during idle time.

    Order of precedence:
    1. ``KAIROS_IDLE_FINETUNE_SYMBOLS`` (space or comma separated)
    2. ``KAIROS_IDLE_FINETUNE_SYMBOL`` for backward compatibility
    3. All distinct tickers in the local price-cache DB (if available)
    4. ``DEFAULT_FINETUNE_SYMBOLS``
    """
    symbols_env = os.environ.get("KAIROS_IDLE_FINETUNE_SYMBOLS")
    if symbols_env:
        return [s.strip() for s in symbols_env.replace(",", " ").split() if s.strip()]
    single_symbol = os.environ.get("KAIROS_IDLE_FINETUNE_SYMBOL")
    if single_symbol:
        return [single_symbol]
    db_symbols = _symbols_from_price_cache_db()
    if db_symbols:
        return db_symbols
    return DEFAULT_FINETUNE_SYMBOLS


def _symbols_from_price_cache_db() -> list[str]:
    """Return all distinct tickers present in the local price cache SQLite DB."""
    import sqlite3

    db_path = REPO_ROOT / "data" / "yfd_prices.db"
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        rows = conn.execute("SELECT DISTINCT ticker FROM prices ORDER BY ticker").fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        conn.close()


def _load_state() -> dict:
    """Load persistent state for the idle runner."""
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    """Persist idle runner state."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _load_last_symbol(symbols: list[str]) -> str | None:
    """Return the last symbol that successfully trained, if it is still in the list."""
    last = _load_state().get("last_symbol")
    if last in symbols:
        return last
    return None


def _record_training(symbol: str) -> None:
    """Persist the last trained symbol and mark `symbol` as just trained."""
    state = _load_state()
    state["last_symbol"] = symbol
    state.setdefault("last_trained", {})[symbol] = datetime.now(timezone.utc).isoformat()
    _save_state(state)


def _is_on_cooldown(symbol: str, cooldown_seconds: float) -> bool:
    """Return True iff `symbol` was trained within the last `cooldown_seconds`."""
    if cooldown_seconds <= 0:
        return False
    trained_at = _load_state().get("last_trained", {}).get(symbol)
    if not trained_at:
        return False
    try:
        last = datetime.fromisoformat(trained_at)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - last).total_seconds() < cooldown_seconds
    except (ValueError, TypeError):
        return False


def rotate_symbols(symbols: list[str], last_symbol: str | None) -> list[str]:
    """Rotate the symbol list so the symbol after `last_symbol` comes first."""
    if not last_symbol or last_symbol not in symbols:
        return symbols
    idx = symbols.index(last_symbol)
    next_idx = (idx + 1) % len(symbols)
    return symbols[next_idx:] + symbols[:next_idx]


def filter_cooldown_symbols(symbols: list[str], cooldown_seconds: float) -> list[str]:
    """Return symbols that are not currently on cooldown."""
    return [s for s in symbols if not _is_on_cooldown(s, cooldown_seconds)]


def build_finetune_command(symbol: str | None = None) -> list[str]:
    """Build the command line for a single-symbol fine-tuning run.

    ``KAIROS_IDLE_FINETUNE_CMD`` overrides everything (a shell-style string,
    so it can include a full pipeline).  Otherwise, uses the environment vars
    ``KAIROS_IDLE_FINETUNE_MODEL`` / ``KAIROS_IDLE_FINETUNE_SYMBOL`` /
    ``KAIROS_IDLE_FINETUNE_OUTPUT``.
    """
    override = os.environ.get("KAIROS_IDLE_FINETUNE_CMD")
    if override:
        return shlex.split(override)

    model = os.environ.get("KAIROS_IDLE_FINETUNE_MODEL", DEFAULT_FINETUNE_MODEL)
    if symbol is None:
        symbol = os.environ.get("KAIROS_IDLE_FINETUNE_SYMBOL", DEFAULT_FINETUNE_SYMBOL)
    output = os.environ.get(
        "KAIROS_IDLE_FINETUNE_OUTPUT",
        str(REPO_ROOT / "models" / "idle_finetune" / f"{model}_{symbol}"),
    )
    uv = shutil.which("uv") or "uv"
    return [uv, "run", "finetune", "--model", model, "--symbol", symbol, "--output-model", output]


def _notify(text: str) -> None:
    try:
        send_telegram(text)
    except OpsError as notify_err:
        logger.error("Failed to send Telegram: %s", notify_err)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fine-tune job during idle GPU time, with Telegram alerts",
    )
    parser.add_argument(
        "--util-threshold",
        type=int,
        default=int(os.environ.get("KAIROS_IDLE_UTIL_THRESHOLD", DEFAULT_GPU_UTIL_THRESHOLD)),
        help=f"Max GPU utilization percent to consider idle (default: {DEFAULT_GPU_UTIL_THRESHOLD})",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=int(os.environ.get("KAIROS_IDLE_SAMPLES", DEFAULT_SAMPLES)),
        help=f"Consecutive idle samples required (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=float(os.environ.get("KAIROS_IDLE_SAMPLE_INTERVAL", DEFAULT_SAMPLE_INTERVAL_SECONDS)),
        help=f"Seconds between utilization samples (default: {DEFAULT_SAMPLE_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--notify-skip",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("KAIROS_IDLE_NOTIFY_SKIP", False),
        help="Send a Telegram message when skipped as too busy (default: silent)",
    )
    parser.add_argument(
        "--no-gpu-recovery",
        action="store_true",
        default=False,
        help="Abort if CUDA is not healthy instead of running the recovery ladder",
    )
    parser.add_argument(
        "--allow-reboot",
        action="store_true",
        default=False,
        help="Allow GPU recovery ladder to reboot the machine (set KAIROS_GPU_ALLOW_REBOOT=1 alternatively)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=float(os.environ.get("KAIROS_IDLE_FINETUNE_COOLDOWN_SECONDS", DEFAULT_FINETUNE_COOLDOWN_SECONDS)),
        help=f"Seconds before re-training the same symbol (default: {DEFAULT_FINETUNE_COOLDOWN_SECONDS})",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging()

    if not is_gpu_idle(
        threshold=args.util_threshold,
        samples=args.samples,
        interval_seconds=args.sample_interval,
    ):
        logger.info(
            "GPU utilization at/above %d%% threshold; skipping fine-tune this cycle",
            args.util_threshold,
        )
        if args.notify_skip:
            _notify("💤 Kairos idle fine-tune: GPU busy, skipped this cycle")
        return 0

    if not lock_is_available():
        logger.info("GPU idle but shared lock is held by another Kairos job; skipping this cycle")
        if args.notify_skip:
            _notify("💤 Kairos idle fine-tune: GPU idle but lock held by another job, skipped")
        return 0

    symbols = get_finetune_symbols()
    symbols = rotate_symbols(symbols, _load_last_symbol(symbols))
    symbols = filter_cooldown_symbols(symbols, args.cooldown)
    if not symbols:
        logger.info(
            "All symbols are within the %.0f-second cooldown window; skipping fine-tune this cycle",
            args.cooldown,
        )
        if args.notify_skip:
            _notify("💤 Kairos idle fine-tune: all symbols recently trained, skipped this cycle")
        return 0

    cmd_override = os.environ.get("KAIROS_IDLE_FINETUNE_CMD")
    if cmd_override:
        # A full command override does not accept a per-symbol argument; run it once.
        symbols = [symbols[0]]

    proc: Optional[subprocess.CompletedProcess] = None
    try:
        with GpuLock():
            require_gpu(allow_recover=not args.no_gpu_recovery, allow_reboot=args.allow_reboot)
            for symbol in symbols:
                cmd = build_finetune_command(symbol)
                logger.info("GPU idle and healthy; starting fine-tune for %s: %s", symbol, " ".join(cmd))
                _notify(f"🟢 Kairos idle fine-tune starting ({symbol}):\n`{' '.join(cmd)}`")
                proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)

                if proc.returncode == 0:
                    logger.info("Fine-tune completed for %s", symbol)
                    _notify(f"✅ Kairos idle fine-tune completed ({symbol}):\n`{' '.join(cmd)}`")
                    _record_training(symbol)
                    return 0

                stderr = (proc.stderr or "").lower()
                stderr_tail = (proc.stderr or "")[-2000:]
                if "no data returned" in stderr or "not enough data" in stderr:
                    logger.warning("No data returned for %s; continuing to next symbol", symbol)
                    _notify(f"⚠️ Kairos idle fine-tune: no data for {symbol}, continuing with next symbol")
                    continue

                logger.error(
                    "Fine-tune failed for %s (exit %d): %s", symbol, proc.returncode, proc.stderr
                )
                _notify(
                    f"❌ Kairos idle fine-tune failed for {symbol} (exit {proc.returncode}):\n"
                    f"```\n{stderr_tail}\n```"
                )
                return proc.returncode
    except OpsError as exc:
        logger.error("GPU/lock error: %s", exc)
        _notify(f"❌ Kairos idle fine-tune GPU/lock error:\n```\n{exc}\n```")
        return 1

    # All symbols skipped due to missing data; no GPU work was actually done.
    logger.warning("All symbols skipped due to missing data: %s", symbols)
    _notify(
        f"🟡 Kairos idle fine-tune: all symbols skipped (no data available):\n"
        f"{', '.join(symbols)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
