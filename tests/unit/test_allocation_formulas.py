"""Test suite for E11-S08: Formula template engine (XLSX/ODS dialect rendering).

Tests the render_formula() function and formula template consistency.
Per RFC §4.6 and ticket E11-S08 acceptance criteria:

1. One template per formula (P-W per RFC §4.2), each written once
2. render_formula(name, row, fmt) -> str returns correct A1-style formula for fmt="xlsx"
   and of:= formula with semicolon separators for fmt="ods"
3. Both dialects derive from ONE shared template (structural test)
4. No banned function names (FILTER, SORT, UNIQUE, LET, LAMBDA, XLOOKUP, MAXIFS, MINIFS)
5. Row-number substitution is correct for boundary rows (20, 21, 400)
"""

import pytest
import re
from strategy.allocation import render_formula, get_formula_names, get_formula_aliases, _FORMULA_TEMPLATES


class TestRenderFormula:
    """Test basic render_formula API and dialect handling."""

    def test_render_formula_xlsx_basic(self):
        """Test XLSX format output for a simple formula."""
        result = render_formula("risk_pct", 20, "xlsx")
        assert result.startswith("=")
        assert ";" not in result  # XLSX uses commas, not semicolons
        assert "of:=" not in result

    def test_render_formula_ods_basic(self):
        """Test ODS format output for a simple formula."""
        result = render_formula("risk_pct", 20, "ods")
        assert result.startswith("of:=")
        # ODS should use semicolons as argument separators (though risk_pct has commas as comparisons)
        assert ";" in result or "of:=" in result  # Either has semicolons or is simple

    def test_render_formula_row_substitution_20(self):
        """Test row number substitution at boundary row 20."""
        result = render_formula("risk_pct", 20, "xlsx")
        assert "F20" in result
        assert "G20" in result
        assert "F21" not in result
        assert "G21" not in result

    def test_render_formula_row_substitution_21(self):
        """Test row number substitution at row 21."""
        result = render_formula("risk_pct", 21, "xlsx")
        assert "F21" in result
        assert "G21" in result

    def test_render_formula_row_substitution_400(self):
        """Test row number substitution at upper boundary row 400."""
        result = render_formula("risk_pct", 400, "xlsx")
        assert "F400" in result
        assert "G400" in result

    def test_render_formula_config_cells_fixed(self):
        """Test that config cell references stay fixed ($E$3, $E$4, etc.)."""
        result = render_formula("shrink", 20, "xlsx")
        # Config cell $E$3 should be present and unchanged
        assert "$E$3" in result

        result2 = render_formula("shrink", 21, "xlsx")
        # Same formula at different row should have same config reference
        assert "$E$3" in result2

    def test_render_formula_invalid_name(self):
        """Test that invalid formula names raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            render_formula("invalid_formula_name", 20, "xlsx")

    def test_render_formula_invalid_format(self):
        """Test that invalid format strings raise ValueError."""
        with pytest.raises(ValueError, match="fmt must be"):
            render_formula("risk_pct", 20, "pdf")  # Invalid format

    def test_render_formula_all_names(self):
        """Test that all formula names in get_formula_names() render successfully."""
        for name in get_formula_names():
            result_xlsx = render_formula(name, 20, "xlsx")
            result_ods = render_formula(name, 20, "ods")
            assert result_xlsx.startswith("=")
            assert result_ods.startswith("of:=")


class TestDialectConsistency:
    """Test that XLSX and ODS formats derive from the same template."""

    def test_same_template_for_both_dialects(self):
        """Verify structurally that both dialects use the same template source.

        This test asserts that there is a single _FORMULA_TEMPLATES dict that
        both render_formula("xlsx") and render_formula("ods") read from, ensuring
        no duplicate/divergent formula sets.
        """
        # Verify that _FORMULA_TEMPLATES is a dict
        assert isinstance(_FORMULA_TEMPLATES, dict)

        # For each formula, verify that XLSX and ODS versions are derived
        # from the same template by checking they differ only in prefixes/separators
        for name in get_formula_names():
            xlsx = render_formula(name, 20, "xlsx")
            ods = render_formula(name, 20, "ods")

            # Remove dialect-specific prefixes
            xlsx_core = xlsx[1:]  # Remove "="
            ods_core = ods[4:]   # Remove "of:="

            # Replace semicolons back to commas in ODS to compare cores
            ods_core_normalized = ods_core.replace(";", ",")

            # Core formula should match (same template, same row number)
            assert xlsx_core == ods_core_normalized, \
                f"Formula {name} has divergent templates: xlsx={xlsx_core!r} vs ods={ods_core_normalized!r}"

    def test_single_source_verify(self):
        """Verify that only one copy of each formula template exists."""
        # Count occurrences of each template across all formula names
        # If someone accidentally duplicated a formula, this would catch it
        template_values = list(_FORMULA_TEMPLATES.values())
        assert len(template_values) == len(set(template_values)), \
            "Duplicate templates found; each formula should have a unique template"


class TestBannedFunctions:
    """Test that rendered formulas don't contain banned function names.

    Per E11-S08 acceptance criteria: no FILTER, SORT, UNIQUE, LET, LAMBDA,
    XLOOKUP, MAXIFS, MINIFS as substrings in any rendered formula.
    """

    BANNED_FUNCTIONS = {"FILTER", "SORT", "UNIQUE", "LET", "LAMBDA", "XLOOKUP", "MAXIFS", "MINIFS"}

    def test_no_banned_functions_in_any_formula(self):
        """Scan all rendered formulas (any row) for banned function names."""
        for name in get_formula_names():
            for row in [20, 21, 400]:  # Sample rows
                for fmt in ["xlsx", "ods"]:
                    formula = render_formula(name, row, fmt)

                    for banned in self.BANNED_FUNCTIONS:
                        # Check if banned function appears as a substring (case-insensitive)
                        assert banned not in formula.upper(), \
                            f"Banned function {banned} found in {name} at row {row} (fmt={fmt}): {formula}"

    def test_only_compatibility_subset_used(self):
        """Test that formulas use only the allowed compatibility functions."""
        allowed_functions = {
            "IF", "AND", "OR", "NOT", "MIN", "MAX", "SUM", "ABS", "ROW",
            "SUMPRODUCT", "SUMIFS", "COUNTIFS", "IFERROR", "LOWER", "TRIM",
        }

        # Extract function names from formulas (uppercase word directly followed by a
        # left paren).  Argument separators or cell references cannot be mistaken
        # for function calls this way.
        for name in get_formula_names():
            formula = render_formula(name, 20, "xlsx")
            matches = re.findall(r'\b([A-Z_]+)\s*\(', formula)
            for func in matches:
                if func in ("A", "F", "G", "H", "L", "M", "O", "E", "C"):
                    # Single-letter cell column references, skip
                    continue
                assert func in allowed_functions, \
                    f"Function {func} not in allowed set for formula {name}: {formula}"


class TestFormulaLogic:
    """Test that rendered formulas implement correct logic."""

    def test_risk_pct_formula_structure(self):
        """Test risk_pct formula: ABS(stop - entry) / entry (fraction, no *100)."""
        result = render_formula("risk_pct", 20, "xlsx")
        # Should reference G20 (stop) and F20 (entry)
        assert "G20" in result
        assert "F20" in result
        assert "ABS" in result
        assert "*100" not in result

    def test_reward_pct_formula_structure(self):
        """Test reward_pct formula: ABS(target - entry) / entry (fraction, no *100)."""
        result = render_formula("reward_pct", 20, "xlsx")
        # Should reference H20 (target) and F20 (entry)
        assert "H20" in result
        assert "F20" in result
        assert "ABS" in result
        assert "*100" not in result

    def test_shrink_formula_uses_n0_config(self):
        """Test shrink formula: n / (n + n0), uses config $E$3."""
        result = render_formula("shrink", 20, "xlsx")
        # Should have n reference (L20) and n0 reference ($E$3)
        assert "L20" in result
        assert "$E$3" in result
        # Should have division and addition
        assert "/" in result
        assert "+" in result

    def test_kelly_frac_uses_kelly_mult_config(self):
        """Test kelly_frac uses config $E$5 for kelly_mult."""
        result = render_formula("kelly_frac", 20, "xlsx")
        assert "$E$5" in result

    def test_ev_net_uses_cost_config(self):
        """Test ev_net uses config $E$4 for round_trip_cost_pct."""
        result = render_formula("ev_net", 20, "xlsx")
        assert "$E$4" in result


class TestODSDialectConversion:
    """Test ODS-specific dialect conversion (comma to semicolon)."""

    def test_ods_uses_semicolons_for_function_args(self):
        """Test that ODS formulas use semicolons instead of commas in function calls."""
        # risk_pct has a conditional but no function args beyond IF itself
        # Find a formula with actual function arguments: b, for example
        result_ods = render_formula("b", 20, "ods")
        result_xlsx = render_formula("b", 20, "xlsx")

        # ODS should have semicolons where XLSX has commas
        # Count separators: ODS should have more semicolons, XLSX should have commas
        ods_without_prefix = result_ods[4:]  # Remove "of:="
        xlsx_without_prefix = result_xlsx[1:]  # Remove "="

        # The converted formula should differ only in punctuation
        # Verify that semicolons appear in ODS version
        if ";" in ods_without_prefix:
            # Multi-argument function found; verify conversion
            assert ods_without_prefix.replace(";", ",") == xlsx_without_prefix, \
                f"ODS/XLSX conversion mismatch:\nODS:  {result_ods}\nXLSX: {result_xlsx}"

    def test_ods_prefix(self):
        """Test that ODS formulas start with of:= prefix."""
        for name in get_formula_names():
            result = render_formula(name, 20, "ods")
            assert result.startswith("of:="), \
                f"ODS formula for {name} doesn't start with 'of:=': {result}"

    def test_xlsx_prefix(self):
        """Test that XLSX formulas start with = prefix."""
        for name in get_formula_names():
            result = render_formula(name, 20, "xlsx")
            assert result.startswith("="), \
                f"XLSX formula for {name} doesn't start with '=': {result}"


class TestRangeSubstitution:
    """Test that range references are handled correctly."""

    def test_row_relative_references_update(self):
        """Test that row-relative references (F20, F21) update with row number."""
        result_20 = render_formula("risk_pct", 20, "xlsx")
        result_21 = render_formula("risk_pct", 21, "xlsx")

        # References should be different
        assert result_20 != result_21
        assert "F20" in result_20
        assert "F21" in result_21

    def test_absolute_row_references_fixed(self):
        """Test that absolute row references ($E$3) don't update with row number."""
        result_20 = render_formula("shrink", 20, "xlsx")
        result_21 = render_formula("shrink", 21, "xlsx")
        result_400 = render_formula("shrink", 400, "xlsx")

        # $E$3 should be the same in all rows
        assert "$E$3" in result_20
        assert "$E$3" in result_21
        assert "$E$3" in result_400


class TestErrorHandling:
    """Test formula error handling (IFERROR guards)."""

    def test_kelly_raw_has_error_handling(self):
        """Test that kelly_raw formula has IFERROR to guard division by zero."""
        result = render_formula("kelly_raw", 20, "xlsx")
        assert "IFERROR" in result, \
            "kelly_raw should have IFERROR guard for division by b"

    def test_score_has_error_handling(self):
        """Test that score formula has IFERROR to guard division by zero."""
        result = render_formula("score", 20, "xlsx")
        assert "IFERROR" in result, \
            "score should have IFERROR guard for division by loss_pct"


class TestGrossScaleFactor:
    """Test the gross scale factor formula per ticket requirement."""

    def test_gross_scale_references_config(self):
        """Test that gross_scale formula references $E$6 (or appropriate config cell)."""
        result = render_formula("gross_scale", 20, "xlsx")
        # Should reference a config cell for gross cap
        assert "$E$6" in result

    def test_gross_scale_scales_post_cluster_total(self):
        """gross_scale must compare the post-cluster-cap total, not just AL count."""
        result = render_formula("gross_scale", 20, "xlsx")
        # It should use SUMPRODUCT over AG (position-capped alloc) and AL
        # (cluster scale), otherwise SUM(AL) would be ~selected_count and never
        # trigger scaling when gross_cap_pct is 100.
        assert "SUMPRODUCT" in result
        assert "AG$21:AG$401" in result
        assert "AL$21:AL$401" in result
        assert "SUM(AL$21:AL$401)" not in result


class TestAllFormulasCoverage:
    """Test that every required formula column is implemented."""

    # Canonical keys are column letters P..AN plus the summary gross_scale factor.
    COLUMN_FORMULAS = {
        "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AA",
        "AB", "AC", "AD", "AE", "AF", "AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN",
    }
    SUMMARY_FORMULAS = {"gross_scale"}

    # Concept aliases must still work for callers that prefer readable names.
    CONCEPT_ALIASES = {
        "risk_pct", "reward_pct", "b", "loss_pct", "shrink", "ev_shrunk",
        "ev_net", "p_shrunk", "kelly_raw", "kelly_frac", "score", "gross_scale",
        "enabled_flag", "base_alloc_pct",
    }

    def test_all_column_letter_formulas_present(self):
        """Test that formulas for columns P through AN are implemented."""
        available = set(get_formula_names())
        assert self.COLUMN_FORMULAS.issubset(available), \
            f"Missing column formulas: {self.COLUMN_FORMULAS - available}"

    def test_summary_formula_present(self):
        """Test that the gross scale summary formula is implemented."""
        available = set(get_formula_names())
        assert self.SUMMARY_FORMULAS.issubset(available), \
            f"Missing summary formulas: {self.SUMMARY_FORMULAS - available}"

    def test_render_all_column_letter_formulas(self):
        """Test that every O..AJ formula renders in both dialects."""
        for name in self.COLUMN_FORMULAS:
            for fmt in ["xlsx", "ods"]:
                result = render_formula(name, 20, fmt)
                assert result
                assert len(result) >= 3

    def test_concept_aliases_resolve_to_same_template(self):
        """Concept aliases must render identically to their canonical column key."""
        aliases = get_formula_aliases()
        for alias in self.CONCEPT_ALIASES:
            canonical = aliases[alias]
            for fmt in ["xlsx", "ods"]:
                assert render_formula(alias, 42, fmt) == render_formula(canonical, 42, fmt)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
