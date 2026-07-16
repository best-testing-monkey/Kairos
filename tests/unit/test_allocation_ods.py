"""Test suite for E11-S10: write_ods_sheet() ODS allocation sheet writer.

Uses an in-memory odfpy document, saves to BytesIO, reloads, and asserts the
sheet layout, formulas, config block, and header row match the XLSX writer.
"""

from io import BytesIO

import pytest
from odf import opendocument
from odf.table import Table, TableCell, TableRow
from odf.text import P

from allocation import (
    AllocationConfig,
    AllocationResult,
    Candidate,
    compute_derived,
    render_formula,
    write_ods_sheet,
    _sorted_for_sheet,
)


def _make_candidate(**kwargs):
    """Build a Candidate with sensible defaults; override via kwargs."""
    defaults = {
        "strategy": "path_execution",
        "ticker": "TEST",
        "direction": "long",
        "entry": 100.0,
        "stop": 95.0,
        "target": 110.0,
        "ev_pct": 5.0,
        "base_win_rate": 0.55,
        "n": 100,
        "backtest_period": "2023-01-01 to 2023-12-31",
        "sharpe": 1.2,
        "advised_liquidity_pct": 10.0,
    }
    defaults.update(kwargs)
    return Candidate(**defaults)


def _make_row(candidate, config, status, flags=None, alloc=None):
    """Construct a result row dict from a Candidate."""
    derived = compute_derived(candidate, config)
    row = {
        "strategy": candidate.strategy,
        "ticker": candidate.ticker,
        "direction": candidate.direction,
        "entry": candidate.entry,
        "stop": candidate.stop,
        "target": candidate.target,
        "ev_pct": candidate.ev_pct,
        "base_win_rate": candidate.base_win_rate,
        "n": candidate.n,
        "backtest_period": candidate.backtest_period,
        "sharpe": candidate.sharpe,
        "advised_liquidity_pct": candidate.advised_liquidity_pct,
        "derived": derived,
        "status": status,
        "flags": list(flags or []),
    }
    if alloc is not None:
        row["alloc"] = alloc
    return row


def _cell_text(cell):
    """Return the displayed text of an odfpy TableCell."""
    ps = list(cell.getElementsByType(P))
    if not ps:
        return ""
    parts = []
    for p in ps:
        node = p.firstChild
        while node is not None:
            if hasattr(node, "data"):
                parts.append(node.data)
            node = node.nextSibling
    return "".join(parts)


def _cell_value(cell):
    """Return the logical value of an odfpy TableCell (formula, number, or text)."""
    formula = cell.getAttribute("formula")
    if formula:
        return formula
    valuetype = cell.getAttribute("valuetype")
    value = cell.getAttribute("value")
    if value not in (None, ""):
        if valuetype == "float":
            return float(value)
        if valuetype == "boolean":
            return value.lower() == "true"
        return value
    return _cell_text(cell)


def _get_rows(document):
    """Return the rows of the Allocation table in the loaded document."""
    table = next(
        t for t in document.spreadsheet.getElementsByType(Table)
        if t.getAttribute("name") == "Allocation"
    )
    return list(table.getElementsByType(TableRow))


@pytest.fixture
def config():
    return AllocationConfig(
        cluster_map={"REMX": "metals_miners", "NG": "energy"},
    )


@pytest.fixture
def result(config):
    """AllocationResult with one selected and one rejected candidate."""
    selected = _make_candidate(
        ticker="REMX", direction="short", entry=79.73, stop=84.51, target=73.71,
        ev_pct=4.04, base_win_rate=0.47, n=161,
    )
    selected_derived = compute_derived(selected, config)
    selected_alloc = min(selected_derived["kelly_frac"] * 100, config.max_pos_pct)

    rejected = _make_candidate(
        ticker="NG", direction="long", entry=3.0, stop=2.97, target=3.14,
        ev_pct=1.45, base_win_rate=0.52, n=109,
    )

    rows = [
        _make_row(selected, config, "SELECTED", ["DATA_MISMATCH"], selected_alloc),
        _make_row(rejected, config, "BELOW_TOPK", []),
    ]
    return AllocationResult(
        rows=rows,
        selected_count=1,
        gross_exposure_pct=selected_alloc,
        rejection_counts={"BELOW_TOPK": 1},
    )


@pytest.fixture
def document(result, config):
    """Return a loaded odfpy document containing the Allocation sheet."""
    doc = opendocument.OpenDocumentSpreadsheet()
    write_ods_sheet(
        doc,
        result,
        config,
        report_date="2026-07-13",
        generator_version="1.0.0-test",
    )
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return opendocument.load(buffer)


class TestSheetStructure:
    """High-level layout checks."""

    def test_sheet_name(self, document):
        names = [
            t.getAttribute("name")
            for t in document.spreadsheet.getElementsByType(Table)
        ]
        assert "Allocation" in names

    def test_title_row(self, document):
        rows = _get_rows(document)
        assert "Portfolio Allocation" in str(_cell_value(rows[0].getElementsByType(TableCell)[0]))
        assert "2026-07-13" in str(_cell_value(rows[0].getElementsByType(TableCell)[1]))
        assert "1.0.0-test" in str(_cell_value(rows[0].getElementsByType(TableCell)[2]))

    def test_unprotected_note(self, document):
        rows = _get_rows(document)
        note = _cell_value(rows[1].getElementsByType(TableCell)[0])
        assert "unprotected" in str(note).lower()

    def test_config_block_layout(self, document, config):
        rows = _get_rows(document)
        # Config block starts at row index 2 (spreadsheet row 3).
        # Label in column D (idx 3), editable value in column E (idx 4),
        # locked default in column F (idx 5).
        assert _cell_value(_get_cell(rows[2], 3)) == "n0"
        assert _cell_value(_get_cell(rows[2], 4)) == float(config.n0)
        assert _cell_value(_get_cell(rows[2], 5)) == 100.0  # shipped default

        assert _cell_value(_get_cell(rows[3], 3)) == "round_trip_cost_pct"
        assert abs(_cell_value(_get_cell(rows[3], 4)) - config.round_trip_cost_pct / 100.0) < 1e-9
        assert abs(_cell_value(_get_cell(rows[3], 5)) - 0.0015) < 1e-9

        assert _cell_value(_get_cell(rows[8], 3)) == "equity"
        equity_value = _cell_value(_get_cell(rows[8], 4))
        assert equity_value == "" or equity_value is None

        # All 11 config rows are present (rows 3 through 13, indices 2-12).
        for idx in range(2, 13):
            assert _cell_value(_get_cell(rows[idx], 3)) is not None
            assert str(_cell_value(_get_cell(rows[idx], 3))) != ""

    def test_summary_block_formulas(self, document):
        rows = _get_rows(document)
        assert _cell_value(_get_cell(rows[13], 0)) == "Selected count"
        assert str(_cell_value(_get_cell(rows[13], 3))).startswith("of:=")
        assert _cell_value(_get_cell(rows[14], 0)) == "Gross exposure %"
        assert str(_cell_value(_get_cell(rows[14], 3))).startswith("of:=")
        assert _cell_value(_get_cell(rows[15], 0)) == "Enabled count"
        assert str(_cell_value(_get_cell(rows[15], 3))).startswith("of:=")
        assert _cell_value(_get_cell(rows[13], 4)) == render_formula("gross_scale", 14, "ods")
        assert _cell_value(_get_cell(rows[16], 0)) == "EV total"
        assert str(_cell_value(_get_cell(rows[16], 3))).startswith("of:=")
        assert "Q21:Q401" in str(_cell_value(_get_cell(rows[16], 3)))

    def test_blank_row_18(self, document):
        rows = _get_rows(document)
        # Row index 17 is the new genuinely-blank spacer row (spreadsheet row 18).
        cells = rows[17].getElementsByType(TableCell)
        for cell in cells:
            assert _cell_value(cell) in (None, "")

    def test_instruction_line(self, document):
        rows = _get_rows(document)
        instruction = _cell_value(_get_cell(rows[18], 0))
        assert instruction is not None
        assert isinstance(instruction, str)
        assert "computed" in instruction
        assert "column A" in instruction


class TestHeaderRow:
    """Header row 20 (index 19) spot checks per RFC §5.2."""

    def test_header_row_static_columns(self, document):
        rows = _get_rows(document)
        header_cells = rows[19].getElementsByType(TableCell)
        assert _cell_value(header_cells[0]) == "Enabled"
        assert _cell_value(header_cells[1]) == "Ticker"
        assert _cell_value(header_cells[2]) == "Cluster"
        assert _cell_value(header_cells[3]) == "Strategy"
        assert _cell_value(header_cells[4]) == "Dir"
        assert _cell_value(header_cells[5]) == "Entry"
        assert _cell_value(header_cells[6]) == "Stop"
        assert _cell_value(header_cells[7]) == "Target"
        assert _cell_value(header_cells[14]) == "EV raw %"

    def test_header_row_formula_columns(self, document):
        rows = _get_rows(document)
        header_cells = rows[19].getElementsByType(TableCell)
        assert _cell_value(header_cells[15]) == "EV net %"
        assert _cell_value(header_cells[16]) == "EV total"
        assert _cell_value(header_cells[19]) == "Alloc %"
        assert _cell_value(header_cells[21]) == "Flags"
        assert _cell_value(header_cells[22]) == "Advised liq % (ignored)"

    def test_header_row_helper_columns(self, document):
        rows = _get_rows(document)
        header_cells = rows[19].getElementsByType(TableCell)
        assert _cell_value(header_cells[25]) == "b"
        assert _cell_value(header_cells[27]) == "shrink"
        assert _cell_value(header_cells[37]) == "cluster_scale"
        assert _cell_value(header_cells[38]) == "enabled_flag"
        assert _cell_value(header_cells[39]) == "base_alloc_pct"


class TestDataRows:
    """Checks for the per-candidate data rows."""

    def test_data_rows_written_in_sorted_order(self, document, result):
        rows = _get_rows(document)
        sorted_rows = _sorted_for_sheet(result.rows)
        assert _cell_value(_get_cell(rows[20], 1)) == sorted_rows[0]["ticker"]
        assert _cell_value(_get_cell(rows[21], 1)) == sorted_rows[1]["ticker"]

    def test_static_values_from_result(self, document, result):
        rows = _get_rows(document)
        row0 = _sorted_for_sheet(result.rows)[0]
        assert _cell_value(_get_cell(rows[20], 0)) == "true"  # SELECTED -> Enabled "true"
        assert _cell_value(_get_cell(rows[20], 1)) == row0["ticker"]
        assert _cell_value(_get_cell(rows[20], 2)) == "metals_miners"
        assert _cell_value(_get_cell(rows[20], 3)) == row0["strategy"]
        assert _cell_value(_get_cell(rows[20], 4)) == "Short"
        assert _cell_value(_get_cell(rows[20], 5)) == row0["entry"]
        assert _cell_value(_get_cell(rows[20], 6)) == row0["stop"]
        assert _cell_value(_get_cell(rows[20], 7)) == row0["target"]
        assert abs(_cell_value(_get_cell(rows[20], 14)) - row0["ev_pct"] / 100.0) < 1e-9

    def test_rejected_row_enabled_defaults_false(self, document, result):
        rows = _get_rows(document)
        row1 = _sorted_for_sheet(result.rows)[1]
        assert row1["status"] == "BELOW_TOPK"
        assert _cell_value(_get_cell(rows[21], 0)) == "false"

    def test_formula_cell_matches_render_formula(self, document):
        rows = _get_rows(document)
        assert _cell_value(_get_cell(rows[20], 15)) == render_formula("P", 21, "ods")
        assert _cell_value(_get_cell(rows[20], 19)) == render_formula("T", 21, "ods")
        assert _cell_value(_get_cell(rows[20], 37)) == render_formula("AL", 21, "ods")
        assert _cell_value(_get_cell(rows[20], 38)) == render_formula("AM", 21, "ods")
        assert _cell_value(_get_cell(rows[20], 39)) == render_formula("AN", 21, "ods")


class TestClusterExposure:
    """Section B cluster-exposure table checks."""

    def test_cluster_header_and_rows(self, document, config):
        rows = _get_rows(document)
        # Data ends at row index 21; blank separator at 22; cluster header at 23.
        assert _cell_value(_get_cell(rows[23], 0)) == "Cluster"
        assert _cell_value(_get_cell(rows[23], 1)) == "Positions"
        assert _cell_value(_get_cell(rows[23], 2)) == "Gross %"
        assert _cell_value(_get_cell(rows[23], 3)) == "Cap %"
        assert _cell_value(_get_cell(rows[23], 4)) == "Capped?"

        clusters = {
            _cell_value(_get_cell(rows[r], 0))
            for r in range(24, 26)
        }
        assert clusters == {"energy", "metals_miners"}


class TestPercentStyle:
    """Percent-labeled cells carry the KairosPercentCell style."""

    def test_percent_cell_style_applied(self, document):
        rows = _get_rows(document)
        # O column (index 14) is a percent-labeled static value (EV raw %).
        cell = _get_cell(rows[20], 14)
        assert cell.getAttribute("stylename") == "KairosPercentCell"

    def test_percent_style_defined_in_automaticstyles(self, document):
        from odf.style import Style
        style_names = {
            s.getAttribute("name")
            for s in document.automaticstyles.getElementsByType(Style)
        }
        assert "KairosPercentCell" in style_names


class TestAutofilter:
    """table:database-range with filter buttons over the header + data range."""

    def test_database_range_present(self, document):
        from odf.table import DatabaseRange

        ranges = document.spreadsheet.getElementsByType(DatabaseRange)
        assert ranges, "expected at least one table:database-range element"
        target = ranges[0].getAttribute("targetrangeaddress")
        assert "Allocation" in target
        assert "A20" in target
        assert ranges[0].getAttribute("displayfilterbuttons") == "true"


class TestAllocPctRedistribution:
    """New column T: live redistribution formula among enabled rows."""

    def test_alloc_pct_formula_structure(self, document):
        rows = _get_rows(document)
        formula = str(_cell_value(_get_cell(rows[20], 19)))
        assert "AM21" in formula
        assert "AN21" in formula
        assert "AN$21:AN$401" in formula
        assert "AM$21:AM$401" in formula
        assert "SUMPRODUCT" in formula


class TestTypeValidation:
    """Input validation for the writer."""

    def test_rejects_non_ods_document(self, result, config):
        with pytest.raises(TypeError):
            write_ods_sheet(
                "not a document", result, config,
                report_date="2026-07-13", generator_version="1.0.0-test",
            )


def _get_cell(row, col_idx):
    """Return the cell at column index ``col_idx`` from an ODS row."""
    cells = list(row.getElementsByType(TableCell))
    if col_idx < len(cells):
        return cells[col_idx]
    return TableCell()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
