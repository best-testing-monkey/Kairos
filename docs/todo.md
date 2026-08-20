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
- [x] E7-S20 Max adverse excursion (per-signal isolated drawdown) pure function (docs/tickets/E7-S20-max-adverse-excursion.md)
- [x] E7-S21 Closure computation pipeline (docs/tickets/E7-S21-closure-computation-pipeline.md)

## Epic 8 — Offline allocation replay loop

- [x] E8-S22 Data-driven replay step grid & per-step candidate loading (docs/tickets/E8-S22-replay-step-grid-and-loading.md)
- [x] E8-S23 Replay loop core, unleveraged (docs/tickets/E8-S23-replay-loop-core.md)

## Epic 9 — CLI & closeout

- [x] E9-S24 CLI wiring: --precompute / --replay (docs/tickets/E9-S24-cli-wiring.md)
- [x] E9-S25 Dedicated cache-reuse & engine_version-bump regression tests (docs/tickets/E9-S25-cache-reuse-tests.md)
- [x] E9-S26 Document non-goals in --help and module docstring (docs/tickets/E9-S26-docs-non-goals.md)
- [x] E9-S27 End-to-end integration smoke test (docs/tickets/E9-S27-integration-smoke-test.md)

---

# Kairos Multi-Interval Rollout (1d → 1h) — Implementation Todo

Ordered by dependency. Check off an item only in the same commit that completes it.
Source design: `docs/tickets/DESIGN_DOC_multi_interval_1h.md`. E0 (shared plumbing
hardening) is already implemented and committed — not re-listed here, see the design
doc §2 for its detail. Stories marked ⚠️ require a GPU and/or live data and must NOT
be run via unattended `/run-stories` automation — execute manually or under
supervision; everything else is safe for normal cheap-model automation.

## Epic 10 — Universe stage for 1h

- [x] E10-S01 Interval-aware ann_vol annualization in compute_universe_stats (docs/tickets/E10-S01-universe-stats-annualization.md)
- [x] E10-S02 Native-interval liquidity fetch in run_stage_universe (docs/tickets/E10-S02-universe-native-interval-fetch.md)
- [x] E10-S03 Interval-scaled liquidity thresholds and min_bars (docs/tickets/E10-S03-universe-interval-scaled-thresholds.md)
- [x] E10-S04 Real interval_probe_ok gate + hourly-universe-screen playbook (docs/tickets/E10-S04-universe-probe-gate-and-playbook.md)

## Epic 11 — Correlation stage for 1h

- [x] E11-S01 Interval-scaled min_overlap/roll_window (docs/tickets/E11-S01-correlation-interval-scaled-windows.md)
- [x] E11-S02 ⚠️ Live-verify correlation for 1h + playbook (docs/tickets/E11-S02-correlation-live-verify-and-playbook.md) — found 2 bugs, see E11-S03/BUG-03
- [x] E11-S03 Fix correlation fetch-window scaling (bars_needed) so 1h actually produces pairs (docs/tickets/E11-S03-correlation-fetch-window-scaling.md)

## Bugs found via live verification (not part of the original E10-E17 scope)

- [x] BUG-03 Fix DST-ambiguous-time crash in hourly local-fallback fetch (docs/tickets/BUG-03-dst-ambiguous-time-hourly-fetch.md)
- [x] BUG-04 (cross-repo — FIXED 2026-08-20 upstream in price_cache commit `72bac58`, propagated via phantom_ledger submodule bump `8f2d087` + `uv lock --upgrade-package price-cache --upgrade-package phantom-ledger` + `uv sync` in Kairos) The SAME DST-ambiguous-time crash existed in the vendored `price_cache` package's own fetch path (the PRIMARY fetch, not Kairos's local fallback that BUG-03 fixed). Root cause was deeper than expected: crypto's two real hourly bars at the Nov 2 fall-back both localized to the same naive wall-clock string, so `INSERT OR REPLACE` silently collapsed them into one row, and separately yfinance returns crypto intraday bars in UTC (not NY like equities) — price_cache was blindly `tz_localize(None)`-ing them, mislabeling every crypto bar by 4-5 hours, not just at the DST edge. Both fixed upstream; schema bumped to v4 there to purge previously-mislabeled cached sub-daily rows. Live re-verified 2026-08-20: `--stage universe --interval 1h` now fetches real crypto data (e.g. `BTC-USD bars=9410`, vs. total fetch-error crash before) with zero DST errors; `MKR-USD` passes legitimately. **Note, separate from this bug and NOT yet investigated:** most crypto symbols report `$vol=0.0` on this run (real data, real ATR%, but zero dollar volume) — worth a follow-up look before trusting 1h universe screening's pass/fail numbers, but it's a distinct question from the DST crash this ticket covered.

## Epic 12 — Oracle stage for 1h + OrchestratorConfig calibration

- [x] E12-S01 Interval-keyed OrchestratorConfig preset mechanism (docs/tickets/E12-S01-orchestrator-config-interval-presets.md)
- [ ] E12-S02 ⚠️ Live debug_filters=True calibration sweep for 1h (docs/tickets/E12-S02-orchestrator-1h-calibration-sweep.md)
- [ ] E12-S03 ⚠️ Live-verify oracle stage for 1h + playbook (docs/tickets/E12-S03-oracle-1h-live-verify-and-playbook.md)

## Epic 13 — Base model backtest for 1h

- [ ] E13-S01 ⚠️ Live-verify base stage for 1h + playbook (docs/tickets/E13-S01-base-model-1h-live-verify-and-playbook.md)

## Epic 14 — Finetuning loop for 1h

- [ ] E14-S01 ⚠️ Live-verify finetune_next for 1h + playbook (docs/tickets/E14-S01-finetune-next-1h-live-verify-and-playbook.md)

## Epic 15 — Signal generation + selection/allocation for 1h

- [ ] E15-S01 ⚠️ Live-verify kairos_signals.py for 1h + update hourly-signals playbook (docs/tickets/E15-S01-signals-1h-live-verify-and-playbook.md)

## Epic 16 — Papertrade/MTM/margin for 1h

- [ ] E16-S01 Once-per-calendar-day financing/MTM guard (docs/tickets/E16-S01-papertrade-once-per-day-financing-guard.md)
- [ ] E16-S02 ⚠️ Live-verify papertrade for 1h + playbook (docs/tickets/E16-S02-papertrade-1h-live-verify-and-playbook.md)

## Epic 17 — Hourly Telegram digest + scheduling

- [ ] E17-S01 New systemd timer for hourly digest (docs/tickets/E17-S01-hourly-digest-systemd-timer.md)
- [ ] E17-S02 ⚠️ Live-verify digest + hand off timer-enable decision (docs/tickets/E17-S02-hourly-digest-live-verify-and-playbook.md)

---

See `docs/tickets/APPENDIX-A-standards.md` for code style, test conventions, and commit rules that apply to every story.
