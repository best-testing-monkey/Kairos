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
        assert _cell_value(_get_cell(rows[2], 2)) == "n0"
        assert _cell_value(_get_cell(rows[2], 3)) == float(config.n0)
        assert _cell_value(_get_cell(rows[2], 4)) == 100.0  # shipped default

        assert _cell_value(_get_cell(rows[3], 2)) == "round_trip_cost_pct"
        assert abs(_cell_value(_get_cell(rows[3], 3)) - config.round_trip_cost_pct / 100.0) < 1e-9
        assert abs(_cell_value(_get_cell(rows[3], 4)) - 0.0015) < 1e-9

        assert _cell_value(_get_cell(rows[8], 2)) == "equity"
        equity_value = _cell_value(_get_cell(rows[8], 3))
        assert equity_value == "" or equity_value is None

        # All 11 config rows are present (rows 3 through 13, indices 2-12).
        for idx in range(2, 13):
            assert _cell_value(_get_cell(rows[idx], 2)) is not None
            assert str(_cell_value(_get_cell(rows[idx], 2))) != ""

    def test_summary_block_formulas(self, document):
        rows = _get_rows(document)
        assert _cell_value(_get_cell(rows[13], 0)) == "Selected count"
        assert str(_cell_value(_get_cell(rows[13], 2))).startswith("of:=")
        assert _cell_value(_get_cell(rows[14], 0)) == "Gross exposure %"
        assert str(_cell_value(_get_cell(rows[14], 2))).startswith("of:=")
        assert _cell_value(_get_cell(rows[15], 0)) == "Enabled count"
        assert str(_cell_value(_get_cell(rows[15], 2))).startswith("of:=")
        assert _cell_value(_get_cell(rows[13], 3)) == render_formula("gross_scale", 14, "ods")
        assert _cell_value(_get_cell(rows[16], 0)) == "EV total"
        assert str(_cell_value(_get_cell(rows[16], 2))).startswith("of:=")
        assert "P20:P400" in str(_cell_value(_get_cell(rows[16], 2)))

    def test_instruction_line(self, document):
        rows = _get_rows(document)
        instruction = _cell_value(_get_cell(rows[17], 0))
        assert instruction is not None
        assert isinstance(instruction, str)
        assert "computed" in instruction


class TestHeaderRow:
    """Header row 19 spot checks per RFC §5.2."""

    def test_header_row_static_columns(self, document):
        rows = _get_rows(document)
        header_cells = rows[18].getElementsByType(TableCell)
        assert _cell_value(header_cells[0]) == "Ticker"
        assert _cell_value(header_cells[1]) == "Cluster"
        assert _cell_value(header_cells[2]) == "Strategy"
        assert _cell_value(header_cells[3]) == "Dir"
        assert _cell_value(header_cells[4]) == "Entry"
        assert _cell_value(header_cells[5]) == "Stop"
        assert _cell_value(header_cells[6]) == "Target"
        assert _cell_value(header_cells[13]) == "EV raw %"

    def test_header_row_formula_columns(self, document):
        rows = _get_rows(document)
        header_cells = rows[18].getElementsByType(TableCell)
        assert _cell_value(header_cells[14]) == "EV net %"
        assert _cell_value(header_cells[15]) == "EV total"
        assert _cell_value(header_cells[18]) == "Alloc %"
        assert _cell_value(header_cells[20]) == "Flags"
        assert _cell_value(header_cells[21]) == "Advised liq % (ignored)"

    def test_header_row_helper_columns(self, document):
        rows = _get_rows(document)
        header_cells = rows[18].getElementsByType(TableCell)
        assert _cell_value(header_cells[24]) == "b"
        assert _cell_value(header_cells[26]) == "shrink"
        assert _cell_value(header_cells[36]) == "cluster_scale"


class TestDataRows:
    """Checks for the per-candidate data rows."""

    def test_data_rows_written_in_sorted_order(self, document, result):
        rows = _get_rows(document)
        sorted_rows = _sorted_for_sheet(result.rows)
        assert _cell_value(_get_cell(rows[19], 0)) == sorted_rows[0]["ticker"]
        assert _cell_value(_get_cell(rows[20], 0)) == sorted_rows[1]["ticker"]

    def test_static_values_from_result(self, document, result):
        rows = _get_rows(document)
        row0 = _sorted_for_sheet(result.rows)[0]
        assert _cell_value(_get_cell(rows[19], 0)) == row0["ticker"]
        assert _cell_value(_get_cell(rows[19], 1)) == "metals_miners"
        assert _cell_value(_get_cell(rows[19], 2)) == row0["strategy"]
        assert _cell_value(_get_cell(rows[19], 3)) == "Short"
        assert _cell_value(_get_cell(rows[19], 4)) == row0["entry"]
        assert _cell_value(_get_cell(rows[19], 5)) == row0["stop"]
        assert _cell_value(_get_cell(rows[19], 6)) == row0["target"]
        assert abs(_cell_value(_get_cell(rows[19], 13)) - row0["ev_pct"] / 100.0) < 1e-9

    def test_formula_cell_matches_render_formula(self, document):
        rows = _get_rows(document)
        assert _cell_value(_get_cell(rows[19], 14)) == render_formula("O", 20, "ods")
        assert _cell_value(_get_cell(rows[19], 18)) == render_formula("S", 20, "ods")
        assert _cell_value(_get_cell(rows[19], 36)) == render_formula("AK", 20, "ods")


class TestClusterExposure:
    """Section B cluster-exposure table checks."""

    def test_cluster_header_and_rows(self, document, config):
        rows = _get_rows(document)
        # Data ends at row index 20; cluster block starts two rows below at index 22.
        assert _cell_value(_get_cell(rows[22], 0)) == "Cluster"
        assert _cell_value(_get_cell(rows[22], 1)) == "Positions"
        assert _cell_value(_get_cell(rows[22], 2)) == "Gross %"
        assert _cell_value(_get_cell(rows[22], 3)) == "Cap %"
        assert _cell_value(_get_cell(rows[22], 4)) == "Capped?"

        clusters = {
            _cell_value(_get_cell(rows[r], 0))
            for r in range(23, 25)
        }
        assert clusters == {"energy", "metals_miners"}


class TestPercentStyle:
    """Percent-labeled cells carry the KairosPercentCell style."""

    def test_percent_cell_style_applied(self, document):
        rows = _get_rows(document)
        # N column (index 13) is a percent-labeled static value (EV raw %).
        cell = _get_cell(rows[19], 13)
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
        assert "A19" in target
        assert ranges[0].getAttribute("displayfilterbuttons") == "true"


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
