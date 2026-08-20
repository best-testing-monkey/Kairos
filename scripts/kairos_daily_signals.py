#!/usr/bin/env python3
"""Daily Kairos signals runner with optional Telegram alerts.

Intended to be triggered by a systemd user timer a few minutes after the
daily bar closes. It:

1. Acquires the shared GPU lock so it does not overlap with discovery,
   paper-trading, or finetuning jobs.
2. Verifies CUDA is healthy (with optional recovery ladder).
3. Runs ``strategy/kairos_signals.py --intervals 1d --xlsx``.
4. Parses the generated markdown report.
5. Sends a Telegram message only when there are actionable selected signals
   or when the run fails. Empty reports are silent by default.
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Repo root is two levels up from this script.
REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT))

from kairos.ops import GpuLock, OpsError, require_gpu, send_telegram, send_telegram_document  # noqa: E402


STATE_DIR = Path.home() / ".local" / "state" / "kairos"
LOG_PATH = STATE_DIR / "daily_signals.log"


logger = logging.getLogger("kairos_daily_signals")


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


def latest_report_path(out_dir: Path) -> Optional[Path]:
    """Return the most recently modified kairos_signals_*.md file."""
    md_files = sorted(
        out_dir.glob("kairos_signals_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return md_files[0] if md_files else None


def parse_selected_signals(report_text: str) -> tuple[int, int]:
    """Parse (selected, total) from the Portfolio Allocation section.

    Returns (0, 0) if the section is missing or not parseable.
    """
    match = re.search(
        r"Selected\s+(\d+)\s+of\s+(\d+)\s+signals",
        report_text,
        re.IGNORECASE,
    )
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def parse_allocation_rows(report_text: str, max_rows: int = 5) -> list[dict]:
    """Return structured allocation rows for the Telegram digest.

    Each dict has ticker/direction/entry/stop/target/ev_net/alloc as
    pass-through strings straight from the report table, plus
    leverage/margin_pct (also pass-through strings, e.g. "10.0x"/"20.0%")
    which are empty strings when the report wasn't leverage-sized
    (max_leverage <= 1.0 -- allocation.write_md_section() leaves those two
    trailing columns blank in that case).
    """
    section_match = re.search(
        r"## Portfolio Allocation\n\n.*?(\n\| Ticker[^|]+\|[^\n]+\|\n\|[-\s|]+\|\n.*?)(?=\n\n|\n##|\Z)",
        report_text,
        re.DOTALL,
    )
    if not section_match:
        return []

    table_text = section_match.group(1)
    rows = []
    for line in table_text.splitlines():
        line = line.strip()
        if not line.startswith("|") or "Ticker" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        # Expected columns: Ticker, Dir, Strategy, Entry, Stop, Target,
        # EV net, Score, Alloc, Model, [Leverage, Margin %]; skip
        # header/delimiter rows. Leverage/Margin % are only present when
        # the report was generated with --max-leverage > 1.0.
        if len(parts) < 10:
            continue
        rows.append({
            "ticker": parts[1],
            "direction": parts[2],
            "entry": parts[4],
            "stop": parts[5],
            "target": parts[6],
            "ev_net": parts[7],
            "alloc": parts[9],
            "leverage": parts[11] if len(parts) > 11 else "",
            "margin_pct": parts[12] if len(parts) > 12 else "",
        })
        if len(rows) >= max_rows:
            break
    return rows


def _parse_trailing_number(s: str) -> Optional[float]:
    """Parse '20.0%' or '10.0x' -> 20.0 / 10.0; None if not parseable."""
    s = s.strip().rstrip("%x")
    try:
        return float(s)
    except ValueError:
        return None


def format_allocation_row(row: dict) -> str:
    """Format one allocation row as an actionable Telegram line.

    When leverage sizing is present: shows margin %/leverage actually
    committed (straight from the report's own Leverage/Margin % columns,
    unchanged) plus entry/TP/SL and EV expressed as return-on-margin (the
    underlying EV net % scaled by leverage -- e.g. a 0.5% EV net at 10x
    leverage is a 5% expected return on the margin actually posted, since
    that margin is a 1/leverage fraction of the notional exposure).

    Falls back to a plain (unleveraged) line when Leverage/Margin % are
    blank, e.g. for a report run at the default --max-leverage 1.0.
    """
    leverage = _parse_trailing_number(row["leverage"])
    ev_net = _parse_trailing_number(row["ev_net"])

    if leverage and row["margin_pct"]:
        ev_str = f"{ev_net * leverage:.0f}%" if ev_net is not None else "n/a"
        return (
            f"{row['margin_pct']} margin {row['leverage']} leverage "
            f"{row['ticker']} {row['direction']} @ {row['entry']} "
            f"TP {row['target']} SL {row['stop']} (ev {ev_str})"
        )

    ev_str = row["ev_net"] if row["ev_net"] else "n/a"
    return (
        f"{row['ticker']} {row['direction']} @ {row['entry']} "
        f"TP {row['target']} SL {row['stop']} (ev {ev_str}) -- {row['alloc']} alloc"
    )


def parse_failure_count(report_text: str) -> int:
    """Count bullets under the ## Failures section."""
    section_match = re.search(r"## Failures\n\n(.*?)(?=\n##|\Z)", report_text, re.DOTALL)
    if not section_match:
        return 0
    return sum(1 for line in section_match.group(1).splitlines() if line.strip().startswith("- "))


def build_success_message(
    report_text: str,
    report_path: Path,
    selected: int,
    total: int,
    failures: int,
) -> str:
    """Build a concise Telegram message for a non-empty daily report."""
    rows = parse_allocation_rows(report_text)

    lines = [
        f"📊 Kairos daily signals: {selected} selected of {total} candidates",
        f"Report: `{report_path.name}`",
    ]
    if rows:
        lines.append("Top allocations:")
        lines.extend(f"- {format_allocation_row(r)}" for r in rows)
    if failures:
        lines.append(f"⚠️ {failures} fetch/prediction failures — see report.")
    return "\n".join(lines)


def run_signals(
    out_dir: Path,
    intervals: list[str],
    xlsx: bool,
    ods: bool,
    signal_selection: Optional[str] = None,
    cluster_map: Optional[str] = None,
    max_leverage: Optional[float] = None,
    margin_utilization: Optional[float] = None,
) -> subprocess.CompletedProcess:
    """Run kairos_signals.py as a subprocess and return its result."""
    cmd = [sys.executable, str(REPO_ROOT / "strategy" / "kairos_signals.py")]
    if intervals:
        cmd.extend(["--intervals", *intervals])
    if xlsx:
        cmd.append("--xlsx")
    if ods:
        cmd.append("--ods")
    if signal_selection:
        cmd.extend(["--signal-selection", signal_selection])
    if cluster_map:
        cmd.extend(["--cluster_map", cluster_map])
    if max_leverage is not None:
        cmd.extend(["--max-leverage", str(max_leverage)])
    if margin_utilization is not None:
        cmd.extend(["--margin-utilization", str(margin_utilization)])

    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Kairos daily signals and send Telegram alerts on action/failure",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        default=["1d"],
        help="Intervals to pass to kairos_signals.py (default: 1d)",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "results"),
        help="Directory where kairos_signals_*.md reports are written",
    )
    parser.add_argument(
        "--xlsx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate .xlsx allocation sheet (default: True)",
    )
    parser.add_argument(
        "--ods",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate .ods allocation sheet",
    )
    parser.add_argument(
        "--signal-selection",
        dest="signal_selection",
        default=None,
        help="Passed through to kairos_signals.py --signal-selection (rule string).",
    )
    parser.add_argument(
        "--cluster_map",
        dest="cluster_map",
        default=None,
        help="Passed through to kairos_signals.py --cluster_map.",
    )
    parser.add_argument(
        "--max-leverage",
        dest="max_leverage",
        type=float,
        default=None,
        help="Passed through to kairos_signals.py --max-leverage.",
    )
    parser.add_argument(
        "--margin-utilization",
        dest="margin_utilization",
        type=float,
        default=None,
        help="Passed through to kairos_signals.py --margin-utilization.",
    )
    parser.add_argument(
        "--notify-empty",
        action="store_true",
        default=False,
        help="Also send a Telegram message when no signals are selected",
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
    args = parser.parse_args(argv)

    configure_logging()
    out_dir = Path(args.out)

    proc: Optional[subprocess.CompletedProcess] = None
    try:
        with GpuLock():
            require_gpu(
                allow_recover=not args.no_gpu_recovery,
                allow_reboot=args.allow_reboot,
            )
            proc = run_signals(
                out_dir, args.intervals, args.xlsx, args.ods,
                signal_selection=args.signal_selection,
                cluster_map=args.cluster_map,
                max_leverage=args.max_leverage,
                margin_utilization=args.margin_utilization,
            )
    except OpsError as exc:
        logger.error("GPU/lock error: %s", exc)
        try:
            send_telegram(f"❌ Kairos daily signals GPU/lock error:\n```\n{exc}\n```", parse_mode=None)
        except OpsError as notify_err:
            logger.error("Failed to send Telegram: %s", notify_err)
        return 1

    if proc is None or proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-2000:] if proc else "GPU/lock error prevented run"
        if proc is not None:
            logger.error("kairos_signals.py failed: %s", proc.stderr)
        try:
            send_telegram(
                f"❌ Kairos daily signals failed (exit {proc.returncode if proc else 'N/A'}):\n"
                f"```\n{stderr_tail}\n```",
                parse_mode=None,
            )
        except OpsError as notify_err:
            logger.error("Failed to send Telegram: %s", notify_err)
        return proc.returncode if proc else 1

    report_path = latest_report_path(out_dir)
    if report_path is None:
        logger.error("No kairos_signals_*.md report found in %s", out_dir)
        try:
            send_telegram("⚠️ Kairos daily signals ran but no report was found.", parse_mode=None)
        except OpsError as notify_err:
            logger.error("Failed to send Telegram: %s", notify_err)
        return 2

    selected, total = parse_selected_signals(report_path.read_text())
    failures = parse_failure_count(report_path.read_text())
    logger.info("Report: %s — selected %d of %d (failures: %d)", report_path, selected, total, failures)

    if selected > 0:
        message = build_success_message(report_path.read_text(), report_path, selected, total, failures)
        try:
            send_telegram_document(report_path)
        except OpsError as notify_err:
            # Best-effort: send the actionable text alert below regardless.
            logger.error("Failed to send Telegram report attachment: %s", notify_err)
        try:
            send_telegram(message, parse_mode=None)
        except OpsError as notify_err:
            logger.error("Failed to send Telegram: %s", notify_err)
            return 3
    elif args.notify_empty:
        try:
            send_telegram_document(report_path)
        except OpsError as notify_err:
            logger.error("Failed to send Telegram report attachment: %s", notify_err)
        try:
            send_telegram(
                f"📊 Kairos daily signals: no actionable signals selected\nReport: `{report_path.name}`",
                parse_mode=None,
            )
        except OpsError as notify_err:
            logger.error("Failed to send Telegram: %s", notify_err)
            return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
