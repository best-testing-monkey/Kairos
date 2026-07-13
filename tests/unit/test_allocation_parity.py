"""LibreOffice headless formula-parity tests for the Allocation sheet.

Per RFC allocation_sheet.md §8 and ticket E11-S13: recalculate the written
XLSX/ODS sheets with ``soffice --headless``, export the resulting values to CSV,
and spot-check them against the pure-Python output of ``allocate()``.

This module skips entirely when LibreOffice is not installed, so the rest of the
unit suite remains usable on machines without ``soffice``.
"""

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from openpyxl import Workbook
from odf import opendocument

from allocation import (
    AllocationConfig,
    Candidate,
    allocate,
    compute_derived,
    write_ods_sheet,
    write_xlsx_sheet,
)


# Skip the whole module if LibreOffice is unavailable.
SOFFICE = shutil.which("soffice") or shutil.which("libreoffice")
if not SOFFICE:
    pytest.skip(
        "LibreOffice (soffice) not found; skipping formula parity tests",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixture: RFC §7 worked-example rows
# ---------------------------------------------------------------------------

def _candidate(ticker, strategy, direction, entry, stop, target, ev_pct,
               base_win_rate, n):
    """Build a Candidate with the mandatory fields populated."""
    return Candidate(
        strategy=strategy,
        ticker=ticker,
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        ev_pct=ev_pct,
        base_win_rate=base_win_rate,
        n=n,
        backtest_period="2023-01-01 to 2023-12-31",
        sharpe=1.0,
        advised_liquidity_pct=10.0,
    )


@pytest.fixture
def rfc7_candidates():
    """Return the five worked-example candidates from RFC §7."""
    return [
        _candidate("NG=F", "close_direction", "long", 2.95, 2.97, 3.14, 1.45, 0.52, 109),
        _candidate("NG=F", "open_gap", "long", 2.95, 2.97, 3.14, 1.45, 0.52, 79),
        _candidate("REMX", "path_execution", "short", 79.73, 84.51, 73.71, 4.04, 0.47, 161),
        _candidate("V", "path_execution", "short", 200.0, 203.2, 203.6, 1.5, 0.48, 319),
        _candidate("CRM", "path_execution", "long", 200.0, 197.0, 204.6, 2.0, 0.55, 79),
    ]


@pytest.fixture
def config():
    """Default AllocationConfig with a static cluster map."""
    return AllocationConfig(
        n0=100,
        min_n=50,
        round_trip_cost_pct=0.15,
        kelly_mult=0.35,
        top_k=12,
        max_pos_pct=15.0,
        max_cluster_pct=25.0,
        gross_cap_pct=100.0,
        cluster_map={"REMX": "metals_miners", "V": "tech", "CRM": "tech"},
    )


# ---------------------------------------------------------------------------
# soffice helpers
# ---------------------------------------------------------------------------

def _recalculate_to_csv(workbook_path: Path) -> list[list[str]]:
    """Recalculate ``workbook_path`` with LibreOffice and return CSV rows."""
    tmpdir = tempfile.mkdtemp()
    try:
        cmd = [
            SOFFICE,
            "--headless",
            "--convert-to",
            'csv:"Text - txt - csv (StarCalc)":44,34,0,1,,,,,,,,-1',
            "--outdir",
            tmpdir,
            str(workbook_path),
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"soffice conversion failed: {result.stderr}\n{result.stdout}"
            )

        csv_path = Path(tmpdir) / (workbook_path.stem + ".csv")
        if not csv_path.exists():
            raise RuntimeError(f"soffice did not produce {csv_path}")

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.reader(f))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _csv_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLibreOfficeParity:
    """Formula parity between allocate() and LibreOffice-recalculated sheets."""

    def _write_and_recalculate(self, candidates, config, enabled_mask, fmt):
        """Allocate, write to temp file, recalculate, return (result, csv_rows)."""
        result = allocate(candidates, config, enabled_mask or {})

        tmpdir = tempfile.mkdtemp()
        try:
            tmpdir = Path(tmpdir)
            if fmt == "xlsx":
                wb = Workbook()
                write_xlsx_sheet(
                    wb,
                    result,
                    config,
                    report_date="2026-07-13",
                    generator_version="parity-test",
                )
                path = tmpdir / "allocation.xlsx"
                wb.save(path)
            else:
                doc = opendocument.OpenDocumentSpreadsheet()
                write_ods_sheet(
                    doc,
                    result,
                    config,
                    report_date="2026-07-13",
                    generator_version="parity-test",
                )
                path = tmpdir / "allocation.ods"
                doc.save(path)

            csv_rows = _recalculate_to_csv(path)
            return result, csv_rows
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _check_static_and_selected(self, result, csv_rows, config, fmt):
        """Spot-check static columns and selected-row allocation values."""
        # Data rows begin at CSV index 19 (spreadsheet row 20).
        data_csv = csv_rows[19:]
        assert len(data_csv) >= len(result.rows), (
            f"{fmt}: expected at least {len(result.rows)} data rows, "
            f"got {len(data_csv)}"
        )

        selected_by_ticker = {
            r["ticker"]: r for r in result.rows if r.get("status") == "SELECTED"
        }

        for offset, row in enumerate(result.rows):
            excel_row = 20 + offset
            csv_row = data_csv[offset]

            # Static columns A-N (indices 0-13) should match the input data.
            assert csv_row[0] == row.get("ticker", ""), f"A{excel_row} ticker mismatch"
            assert csv_row[2] == row.get("strategy", ""), f"C{excel_row} strategy mismatch"
            assert csv_row[4] == str(row.get("entry", "")), f"E{excel_row} entry mismatch"

            # Column R (index 17) is the final Alloc %. For selected rows it must
            # match the Python alloc within tolerance. Rejected rows have no
            # Python alloc to compare against because selection is computed in
            # Python, not in the sheet formulas.
            ticker = row.get("ticker", "")
            if ticker in selected_by_ticker:
                py_alloc = selected_by_ticker[ticker].get("alloc")
                calc_alloc = _csv_float(csv_row[17])
                assert py_alloc is not None and calc_alloc is not None, (
                    f"R{excel_row}: missing alloc value"
                )
                assert abs(py_alloc - calc_alloc) <= 1e-6, (
                    f"R{excel_row}: expected alloc {py_alloc}, got {calc_alloc}"
                )

    @pytest.mark.parametrize("fmt", ["xlsx", "ods"])
    def test_all_enabled_mask(self, rfc7_candidates, config, fmt):
        """All-enabled mask: conversion succeeds and selected alloc values match."""
        result, csv_rows = self._write_and_recalculate(
            rfc7_candidates, config, {}, fmt
        )
        self._check_static_and_selected(result, csv_rows, config, fmt)

    @pytest.mark.parametrize("ticker", ["NG=F", "REMX", "V", "CRM"])
    @pytest.mark.parametrize("fmt", ["xlsx", "ods"])
    def test_single_signal_disabled(self, rfc7_candidates, config, ticker, fmt):
        """Disable exactly one ticker at a time and verify parity."""
        enabled_mask = {c.ticker: True for c in rfc7_candidates}
        enabled_mask[ticker] = False
        result, csv_rows = self._write_and_recalculate(
            rfc7_candidates, config, enabled_mask, fmt
        )
        self._check_static_and_selected(result, csv_rows, config, fmt)

    @pytest.mark.parametrize("fmt", ["xlsx", "ods"])
    def test_random_masks(self, rfc7_candidates, config, fmt):
        """100 seeded random enabled/disabled masks."""
        import random
        rng = random.Random(42)
        tickers = list({c.ticker for c in rfc7_candidates})

        for _ in range(100):
            enabled_mask = {t: rng.choice([True, False]) for t in tickers}
            result, csv_rows = self._write_and_recalculate(
                rfc7_candidates, config, enabled_mask, fmt
            )
            self._check_static_and_selected(result, csv_rows, config, fmt)
