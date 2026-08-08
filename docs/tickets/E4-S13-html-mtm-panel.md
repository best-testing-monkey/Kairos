# E4-S13 — Add MTM panel to HTML report

**Goal:** Extend `write_html_report()` in `strategy/kairos_papertrade.py` with a second panel showing MTM equity, drawdown shading, margin utilization, and liquidation markers.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.6.
- Read `strategy/kairos_papertrade.py` `write_html_report()` (around line 1195) and `compute_final_metrics()` (output of E4-S12).
- Read `strategy/kairos_mtm.py` `DailySnapshot` (output of E2-S03).
- Modify `strategy/kairos_papertrade.py`.

**Acceptance criteria:**
- [ ] `write_html_report()` accepts an optional `mtm_curve` argument (list of `DailySnapshot`-like objects or a DataFrame read from `kairos_mtm_daily`).
- [ ] Report layout becomes a 3-row subplot: equity curves, margin utilization, metrics table.
- [ ] Top subplot shows both the closed-trade equity curve and the MTM equity curve.
- [ ] Drawdown shading is added under the MTM equity curve (fill between equity and running peak).
- [ ] A margin-utilization line is plotted on the second subplot with a horizontal reference line at the utilization cap.
- [ ] Vertical markers are drawn at dates where `liquidations > 0`.
- [ ] When no MTM data is present (legacy run), the report degrades gracefully to the existing two-row layout.
- [ ] A unit test asserts the generated HTML contains the strings `"MTM equity"`, `"Margin utilization"`, and `"Liquidation"`.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] HTML unit test passes.
- [ ] Changes committed and `docs/todo.md` E4-S13 item checked off.
