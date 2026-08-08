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
- [x] E4-S09 Maintain corrected cash and persist daily MTM snapshots (docs/tickets/E4-S09-corrected-cash-mtm-persist.md)
- [x] E4-S10 Wire admission check and locked-margin order semantics (docs/tickets/E4-S10-admission-orders.md)
- [x] E4-S11 Implement liquidation execution path (docs/tickets/E4-S11-liquidation-execution.md)
- [x] E4-S12 Add MTM metrics block and extend reconciliation (docs/tickets/E4-S12-mtm-metrics.md)
- [x] E4-S13 Add MTM panel to HTML report (docs/tickets/E4-S13-html-mtm-panel.md)

## Epic 5 — Validation & closeout

- [x] E5-S14 Frozen-fixture MTM repro test (docs/tickets/E5-S14-frozen-fixture-mtm-repro.md)
- [x] E5-S15 Leverage-off regression and exposure-cap verification (docs/tickets/E5-S15-leverage-off-regression.md)
- [x] E5-S16 Smoke test, docs update, and ticket closure (docs/tickets/E5-S16-smoke-docs-closeout.md)

---

# Kairos Offline Signal Replay — Implementation Todo

Ordered by dependency. Check off an item only in the same commit that completes it.
Source design: `docs/tickets/DESIGN_DOC_offline_signal_replay.md`. **Unleveraged only** —
see that document's §1/§4 for the explicit phase-scope boundary.

## Epic 6 — Schema & signal ingestion

- [x] E6-S17 Create papertrade_signals / papertrade_signals_closure schema (docs/tickets/E6-S17-signal-replay-schema.md)
- [x] E6-S18 Populate papertrade_signals by unpacking signals_cache (docs/tickets/E6-S18-precompute-unpack-signals-cache.md)

## Epic 7 — Interval selection & closure computation

- [x] E7-S19 Interval-ladder resolution & disqualification (docs/tickets/E7-S19-interval-ladder-disqualification.md)
- [ ] E7-S20 Max adverse excursion (per-signal isolated drawdown) pure function (docs/tickets/E7-S20-max-adverse-excursion.md)
- [ ] E7-S21 Closure computation pipeline (docs/tickets/E7-S21-closure-computation-pipeline.md)

## Epic 8 — Offline allocation replay loop

- [ ] E8-S22 Data-driven replay step grid & per-step candidate loading (docs/tickets/E8-S22-replay-step-grid-and-loading.md)
- [ ] E8-S23 Replay loop core, unleveraged (docs/tickets/E8-S23-replay-loop-core.md)

## Epic 9 — CLI & closeout

- [ ] E9-S24 CLI wiring: --precompute / --replay (docs/tickets/E9-S24-cli-wiring.md)
- [ ] E9-S25 Dedicated cache-reuse & engine_version-bump regression tests (docs/tickets/E9-S25-cache-reuse-tests.md)
- [ ] E9-S26 Document non-goals in --help and module docstring (docs/tickets/E9-S26-docs-non-goals.md)
- [ ] E9-S27 End-to-end integration smoke test (docs/tickets/E9-S27-integration-smoke-test.md)

---

See `docs/tickets/APPENDIX-A-standards.md` for code style, test conventions, and commit rules that apply to every story.
