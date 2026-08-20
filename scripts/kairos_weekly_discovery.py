#!/usr/bin/env python3
"""Weekly Kairos strategy-discovery runner with optional Telegram alerts.

Implements the two-stage discovery pass from
``docs/playbooks/weekly-strategy-discovery.md``:

1. ``kairos_pipeline.py --stage auto --intervals 1d --backtest_period 6m``
2. ``kairos_pipeline.py --stage auto --intervals 1h --backtest_period 3m --skip_universe``

The runner:

* Acquires the shared GPU lock.
* Verifies CUDA health (with optional recovery).
* Runs the two passes sequentially; the second pass reuses the universe/correlation
  output from the first.
* Sends a Telegram summary with runtime and CSV output paths, or a failure alert.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Repo root is two levels up from this script.
REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

from kairos.ops import GpuLock, OpsError, require_gpu, send_telegram  # noqa: E402


STATE_DIR = Path.home() / ".local" / "state" / "kairos"
LOG_PATH = STATE_DIR / "weekly_discovery.log"


logger = logging.getLogger("kairos_weekly_discovery")


class PipelineStage:
    """One stage of the discovery pipeline."""

    def __init__(self, label: str, *args: str):
        self.label = label
        self.args = list(args)
        self.returncode: Optional[int] = None
        self.duration_seconds: float = 0.0
        self.stdout_tail: str = ""
        self.stderr_tail: str = ""
        self.csv_path: Optional[Path] = None


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


def find_latest_csv(prefix: str, results_dir: Path) -> Optional[Path]:
    """Find the most recently modified CSV starting with ``prefix``."""
    candidates = sorted(
        results_dir.glob(f"{prefix}*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def parse_viability_csv(csv_path: Path) -> dict[str, int]:
    """Count viable strategies per interval from a viability CSV."""
    counts: dict[str, int] = {}
    header: list[str] = []
    try:
        with csv_path.open() as fh:
            for i, line in enumerate(fh):
                parts = [p.strip() for p in line.split(",")]
                if i == 0:
                    header = parts
                    continue
                if not parts:
                    continue
                row = dict(zip(header, parts))
                interval = row.get("interval")
                viable = row.get("viable", "").lower() in ("1", "true", "yes")
                if interval:
                    counts.setdefault(interval, 0)
                    if viable:
                        counts[interval] += 1
    except Exception as exc:
        logger.warning("Could not parse viability CSV %s: %s", csv_path, exc)
    return counts


def run_pipeline_stage(stage: PipelineStage, cwd: Path) -> bool:
    """Run one pipeline stage. Returns True on success."""
    cmd = [sys.executable, str(cwd / "strategy" / "kairos_pipeline.py"), *stage.args]
    logger.info("[%s] Running: %s", stage.label, " ".join(cmd))
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    stage.duration_seconds = time.time() - start
    stage.returncode = proc.returncode
    stage.stdout_tail = (proc.stdout or "")[-2000:]
    stage.stderr_tail = (proc.stderr or "")[-2000:]
    logger.info(
        "[%s] Finished in %.1fs with exit code %d",
        stage.label,
        stage.duration_seconds,
        proc.returncode,
    )
    return proc.returncode == 0


def build_failure_message(stages: list[PipelineStage]) -> str:
    lines = ["❌ Kairos weekly discovery failed"]
    for stage in stages:
        status = "✅ OK" if stage.returncode == 0 else f"❌ exit {stage.returncode}"
        lines.append(f"{stage.label}: {status} ({stage.duration_seconds / 60:.1f} min)")
        if stage.returncode != 0:
            tail = stage.stderr_tail or stage.stdout_tail
            lines.append(f"```\n{tail}\n```")
            break
    return "\n".join(lines)


def build_success_message(stages: list[PipelineStage], results_dir: Path) -> str:
    lines = ["🔬 Kairos weekly discovery complete"]
    for stage in stages:
        lines.append(f"{stage.label}: {stage.duration_seconds / 60:.1f} min")

    csv_path = find_latest_csv("auto_viability_report", results_dir)
    if csv_path:
        counts = parse_viability_csv(csv_path)
        if counts:
            lines.append("Viable strategies:")
            for interval, count in sorted(counts.items()):
                lines.append(f"- {interval}: {count}")
        lines.append(f"CSV: `{csv_path.name}`")

    disabled_csv = find_latest_csv("oracle_disabled_strategies", results_dir)
    if disabled_csv:
        lines.append(f"Disabled diff: `{disabled_csv.name}`")

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Kairos weekly strategy discovery and send Telegram alerts",
    )
    parser.add_argument(
        "--include-hourly",
        action="store_true",
        default=False,
        help="Also run the 1h/3m discovery pass (default: daily 1d/6m only)",
    )
    parser.add_argument(
        "--results",
        default=str(REPO_ROOT / "results"),
        help="Directory where pipeline CSVs are written",
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
        help="Allow GPU recovery ladder to reboot the machine",
    )
    args = parser.parse_args(argv)

    configure_logging()
    results_dir = Path(args.results)

    stages = [
        PipelineStage(
            "1d/6m discovery",
            "--stage", "auto",
            "--intervals", "1d",
            "--backtest_period", "6m",
        ),
    ]
    if args.include_hourly:
        stages.append(
            PipelineStage(
                "1h/3m discovery",
                "--stage", "auto",
                "--intervals", "1h",
                "--backtest_period", "3m",
                "--skip_universe",
            ),
        )

    try:
        with GpuLock():
            require_gpu(
                allow_recover=not args.no_gpu_recovery,
                allow_reboot=args.allow_reboot,
            )
            for stage in stages:
                if not run_pipeline_stage(stage, REPO_ROOT):
                    break
    except OpsError as exc:
        logger.error("GPU/lock error: %s", exc)
        try:
            send_telegram(f"❌ Kairos weekly discovery GPU/lock error:\n```\n{exc}\n```", parse_mode=None)
        except OpsError as notify_err:
            logger.error("Failed to send Telegram: %s", notify_err)
        return 1

    failed = [s for s in stages if s.returncode != 0]
    if failed:
        message = build_failure_message(stages)
        try:
            send_telegram(message, parse_mode=None)
        except OpsError as notify_err:
            logger.error("Failed to send Telegram: %s", notify_err)
            return 3
        return failed[0].returncode or 1

    message = build_success_message(stages, results_dir)
    try:
        send_telegram(message, parse_mode=None)
    except OpsError as notify_err:
        logger.error("Failed to send Telegram: %s", notify_err)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
