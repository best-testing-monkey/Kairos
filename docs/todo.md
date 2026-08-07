# Kairos MTM Margin + Leverage — Implementation Todo

Ordered by dependency. Check off an item only in the same commit that completes it.

## Epic 1 — Margin config & asset classification

- [x] E1-S01 Create default IBKR retail margin config file (docs/tickets/E1-S01-margin-config-yaml.md)
- [x] E1-S02 Build margin config loader and symbol classifier (docs/tickets/E1-S02-kairos-margin-module.md)

## Epic 2 — Pure MTM math

- [x] E2-S03 Build daily MTM snapshot dataclasses and pure math (docs/tickets/E2-S03-mtm-snapshot-core.md)
- [x] E2-S04 Add margin admission check (docs/tickets/E2-S04-admission-check.md)
- [x] E2-S05 Add liquidation check (docs/tickets/E2-S05-liquidation-check.md)
- [x] E2-S06 Add financing and borrow-cost accrual (docs/tickets/E2-S06-financing-accrual.md)

## Epic 3 — Allocation config extensions

- [x] E3-S07 Extend AllocationConfig with leverage fields (docs/tickets/E3-S07-allocation-config-leverage.md)

## Epic 4 — Papertrade engine wiring

- [x] E4-S08 Add CLI flags and load margin config in main (docs/tickets/E4-S08-cli-config-load.md)
- [ ] E4-S09 Maintain corrected cash and persist daily MTM snapshots (docs/tickets/E4-S09-corrected-cash-mtm-persist.md)
- [ ] E4-S10 Wire admission check and locked-margin order semantics (docs/tickets/E4-S10-admission-orders.md)
- [ ] E4-S11 Implement liquidation execution path (docs/tickets/E4-S11-liquidation-execution.md)
- [ ] E4-S12 Add MTM metrics block and extend reconciliation (docs/tickets/E4-S12-mtm-metrics.md)
- [ ] E4-S13 Add MTM panel to HTML report (docs/tickets/E4-S13-html-mtm-panel.md)

## Epic 5 — Validation & closeout

- [ ] E5-S14 Frozen-fixture MTM repro test (docs/tickets/E5-S14-frozen-fixture-mtm-repro.md)
- [ ] E5-S15 Leverage-off regression and exposure-cap verification (docs/tickets/E5-S15-leverage-off-regression.md)
- [ ] E5-S16 Smoke test, docs update, and ticket closure (docs/tickets/E5-S16-smoke-docs-closeout.md)

---

See `docs/tickets/APPENDIX-A-standards.md` for code style, test conventions, and commit rules that apply to every story.
