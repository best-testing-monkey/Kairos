"""test_allocation_md.py — Unit tests for allocation.write_md_section().

Verifies the RFC §6 Markdown "Portfolio Allocation" section renderer produces
all required lines in the correct order, using a hand-built AllocationResult.
"""

from allocation import AllocationResult, AllocationConfig, write_md_section


class TestWriteMdSection:
    """Tests for write_md_section() markdown renderer."""

    def _make_config(self) -> AllocationConfig:
        """Return an AllocationConfig with deterministic, non-default values."""
        return AllocationConfig(
            n0=100,
            min_n=50,
            round_trip_cost_pct=0.15,
            kelly_mult=0.35,
            top_k=12,
            max_pos_pct=15,
            max_cluster_pct=25,
            gross_cap_pct=100,
            cluster_map={
                "NG=F": "energy_commodities",
                "REMX": "metals_miners",
                "V": "healthcare",
            },
        )

    def _make_result(self) -> AllocationResult:
        """Return a hand-built AllocationResult with selected and rejected rows.

        Selected rows are already sorted by score descending, matching the
        contract from allocate().
        """
        return AllocationResult(
            rows=[
                {
                    "ticker": "NG=F",
                    "direction": "long",
                    "strategy": "close_direction",
                    "entry": 2.95,
                    "stop": 2.90,
                    "target": 3.14,
                    "derived": {"ev_net": 0.61, "score": 1.01},
                    "status": "SELECTED",
                    "alloc": 13.9,
                    "flags": [],
                },
                {
                    "ticker": "V",
                    "direction": "long",
                    "strategy": "trend_following",
                    "entry": 230.0,
                    "stop": 226.0,
                    "target": 234.0,
                    "derived": {"ev_net": 0.68, "score": 0.43},
                    "status": "SELECTED",
                    "alloc": 2.4,
                    "flags": [],
                },
                {
                    "ticker": "REMX",
                    "direction": "short",
                    "strategy": "path_execution",
                    "entry": 79.73,
                    "stop": 84.51,
                    "target": 73.71,
                    "derived": {"ev_net": 2.34, "score": 0.39},
                    "status": "SELECTED",
                    "alloc": 2.5,
                    "flags": ["DATA_MISMATCH"],
                },
                {
                    "ticker": "CRM",
                    "direction": "long",
                    "strategy": "momentum",
                    "entry": 150.0,
                    "stop": 147.5,
                    "target": 153.5,
                    "derived": {"ev_net": 0.27, "score": 0.18},
                    "status": "BELOW_TOPK",
                    "alloc": 0.0,
                    "flags": [],
                },
                {
                    "ticker": "BTC",
                    "direction": "long",
                    "strategy": "crypto_breakout",
                    "entry": 50000.0,
                    "stop": 49000.0,
                    "target": 51000.0,
                    "derived": {"ev_net": -0.50, "score": -0.10},
                    "status": "NEG_EV_NET",
                    "alloc": 0.0,
                    "flags": [],
                },
                {
                    "ticker": "ETH",
                    "direction": "long",
                    "strategy": "crypto_breakout",
                    "entry": 3000.0,
                    "stop": 2950.0,
                    "target": 3050.0,
                    "derived": {"ev_net": 0.50, "score": 0.20},
                    "status": "LOW_N",
                    "alloc": 0.0,
                    "flags": [],
                },
            ],
            selected_count=3,
            gross_exposure_pct=18.8,
            rejection_counts={"BELOW_TOPK": 1, "LOW_N": 1, "NEG_EV_NET": 1},
        )

    def test_heading_and_config_summary(self):
        """Output starts with heading and config summary line."""
        config = self._make_config()
        result = self._make_result()
        output = write_md_section(result, config)

        assert output.startswith("## Portfolio Allocation")
        assert "Config: n0=100 min_n=50 cost=0.15% kelly_mult=0.35 top_k=12 " in output
        assert "max_pos=15% max_cluster=25% gross_cap=100%" in output

    def test_selection_summary_line(self):
        """Selection summary reports selected count and gross exposure."""
        config = self._make_config()
        result = self._make_result()
        output = write_md_section(result, config)

        assert "Selected 3 of 6 signals. Gross exposure: 18.8%. EV total: 0.00%." in output

    def test_table_header_and_selected_rows(self):
        """Markdown table contains expected header and all selected rows."""
        config = self._make_config()
        result = self._make_result()
        output = write_md_section(result, config)

        assert "| Ticker | Dir   | Strategy        |  Entry |   Stop | Target | EV net | Score | Alloc |" in output
        assert "| NG=F" in output
        assert "| V" in output
        assert "| REMX" in output

    def test_selected_rows_preserve_input_order(self):
        """Selected rows are emitted in input order (already score-sorted)."""
        config = self._make_config()
        result = self._make_result()
        output = write_md_section(result, config)

        ng_pos = output.find("| NG=F")
        v_pos = output.find("| V")
        remx_pos = output.find("| REMX")

        assert 0 < ng_pos < v_pos < remx_pos

    def test_rejected_rows_absent_from_table(self):
        """Rejected tickers do not appear as table data rows."""
        config = self._make_config()
        result = self._make_result()
        output = write_md_section(result, config)

        # CRM, BTC, ETH are rejected; they should not appear after the header
        # as data rows.  Searching for the bare ticker token is ambiguous,
        # so we look for the table-row prefix pattern used by format_table.
        assert "| CRM |" not in output
        assert "| BTC |" not in output
        assert "| ETH |" not in output

    def test_cluster_exposure_line(self):
        """Cluster exposure sums allocations per cluster among selected rows."""
        config = self._make_config()
        result = self._make_result()
        output = write_md_section(result, config)

        assert "Cluster exposure: energy_commodities 13.9%, metals_miners 2.5%, healthcare 2.4%" in output

    def test_rejected_summary_line(self):
        """Rejection summary lists total and counts sorted descending by count."""
        config = self._make_config()
        result = self._make_result()
        output = write_md_section(result, config)

        assert "Rejected: 3 total -- BELOW_TOPK 1, LOW_N 1, NEG_EV_NET 1" in output

    def test_full_snapshot_matches_expected(self):
        """Full rendered section matches the expected snapshot."""
        config = self._make_config()
        result = self._make_result()
        output = write_md_section(result, config)

        expected = """## Portfolio Allocation

Config: n0=100 min_n=50 cost=0.15% kelly_mult=0.35 top_k=12 max_pos=15% max_cluster=25% gross_cap=100%

Selected 3 of 6 signals. Gross exposure: 18.8%. EV total: 0.00%.

| Ticker | Dir   | Strategy        |  Entry |   Stop | Target | EV net | Score | Alloc |
| ------ | ----- | --------------- | ------ | ------ | ------ | ------ | ----- | ----- |
| NG=F   | Long  | close_direction |   2.95 |   2.90 |   3.14 |  0.61% |  1.01 | 13.9% |
| V      | Long  | trend_following | 230.00 | 226.00 | 234.00 |  0.68% |  0.43 |  2.4% |
| REMX   | Short | path_execution  |  79.73 |  84.51 |  73.71 |  2.34% |  0.39 |  2.5% |

Cluster exposure: energy_commodities 13.9%, metals_miners 2.5%, healthcare 2.4%

Rejected: 3 total -- BELOW_TOPK 1, LOW_N 1, NEG_EV_NET 1"""

        assert output == expected

    def test_empty_selected_table_renders_header_only(self):
        """When no rows are selected, the table still has headers and no data rows."""
        config = AllocationConfig()
        result = AllocationResult(
            rows=[
                {
                    "ticker": "X",
                    "direction": "long",
                    "strategy": "s",
                    "entry": 1.0,
                    "stop": 0.9,
                    "target": 1.1,
                    "derived": {"ev_net": -1.0, "score": -0.5},
                    "status": "NEG_EV_NET",
                    "alloc": 0.0,
                    "flags": [],
                },
            ],
            selected_count=0,
            gross_exposure_pct=0.0,
            rejection_counts={"NEG_EV_NET": 1},
        )
        output = write_md_section(result, config)

        assert "Selected 0 of 1 signals. Gross exposure: 0.0%." in output
        assert "| Ticker | Dir | Strategy | Entry | Stop | Target | EV net | Score | Alloc |" in output
        assert "Cluster exposure: none" in output
        assert "Rejected: 1 total -- NEG_EV_NET 1" in output
