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
- [x] BUG-04 (cross-repo — FIXED 2026-08-20 upstream in price_cache commit `72bac58`, propagated via phantom_ledger submodule bump `8f2d087` + `uv lock --upgrade-package price-cache --upgrade-package phantom-ledger` + `uv sync` in Kairos) The SAME DST-ambiguous-time crash existed in the vendored `price_cache` package's own fetch path (the PRIMARY fetch, not Kairos's local fallback that BUG-03 fixed). Root cause was deeper than expected: crypto's two real hourly bars at the Nov 2 fall-back both localized to the same naive wall-clock string, so `INSERT OR REPLACE` silently collapsed them into one row, and separately yfinance returns crypto intraday bars in UTC (not NY like equities) — price_cache was blindly `tz_localize(None)`-ing them, mislabeling every crypto bar by 4-5 hours, not just at the DST edge. Both fixed upstream; schema bumped to v4 there to purge previously-mislabeled cached sub-daily rows. Live re-verified 2026-08-20: `--stage universe --interval 1h` now fetches real crypto data (e.g. `BTC-USD bars=9410`, vs. total fetch-error crash before) with zero DST errors; `MKR-USD` passes legitimately.
- [x] BUG-05 (FIXED 2026-08-21, commit `20bfd0e`) Follow-up from BUG-04's residual note: most crypto symbols reported `$vol=0.0` in 1h universe screening. Root-caused to two stacked, independent issues in `compute_universe_stats` (`strategy/kairos_pipeline.py`), reproduced live against raw yfinance output with price_cache entirely out of the path: (1) yfinance omits volume on ~50% of individual 1h crypto bars even for BTC-USD/ETH-USD, collapsing a flat per-bar median to exactly 0; (2) yfinance's crypto `volume` column is already dollar-denominated, unlike equity/fx which are share/contract-denominated — the pre-existing `close * volume` math was squaring the price for crypto specifically, which had been invisible at 1d (the inflated number always cleared the $10M threshold trivially) and was separately masked at 1h by issue (1)'s zero-collapse, only surfacing once both were fixed together. Fixed by resampling to daily buckets and summing before taking the median across nonzero days (fixes 1), and skipping the close multiplier for `asset_class="crypto"` (fixes 2). Live-verified 2026-08-21: `BTC-USD $vol=8552878080.0` (~$8.5B, plausible) and `ETH-USD $vol=6016311296.0` (~$6.0B, plausible); universe pass count rose 30→41/153 at `--interval 1h` as previously-mis-scored majors (BNB, XLM, HBAR, LDO, SHIB, WLD) now correctly clear the liquidity gate.
- [x] BUG-06 (FIXED 2026-08-21) `run_stage_finetune_next`'s "〰️ backtesting finetuned model" `_notify()` call hardcoded `enabled=True` instead of `enabled=notify`, so it fired for real regardless of `--no-telegram` — the same root cause as Baz getting live Telegram pings whenever a test run happened to exercise that code path with real credentials in the shell. One-line fix (`enabled=notify`), caught the failing `test_no_telegram_flag_suppresses_all_notifications` test.
- [x] BUG-07 (FIXED 2026-08-21) `select_finetune_candidate`'s auto-select ranking had no interval filter, and `run_stage_finetune_next` never forwarded its own `interval` argument into it — so `--interval 1h` had zero effect on auto-select, which would always prefer 1d profiles (far more accumulated oracle history) over 1h ones. Every 1h `finetuned_models` row before this fix was necessarily created via explicit `--assets` manual re-queue, never natural auto-select. Fixed by adding an `interval` parameter to `select_finetune_candidate` and threading it through.
- [x] BUG-08 (FIXED 2026-08-21) Rejected finetuned checkpoints were never deleted from disk — only an empty `REJECTED` marker file was written, but the ~780MB `best_model`/`final_model` weight directories were left in place forever. Found 23 accumulated rejected checkpoints (18 pre-existing at 1d, 5 from today's 1h batch) wasting ~18GB total. Fixed in `run_stage_finetune_next`'s reject branch: delete the weight subdirectories (keep `model_dir`, `metadata.json`, and the `REJECTED` marker for post-mortem), and null out the now-dangling `model_path` in both the registry row and `metadata.json`. Existing 23 rejected directories cleaned up retroactively (weights deleted, registry/metadata `model_path` nulled). See `PREDCACHE-01-finetune-verification-persistent-cache.md` for a related but separate follow-up (purging a rejected model's *cached predictions*, not just its weights, once verification predictions are cached at all — they currently aren't, see that ticket).
- [x] BUG-09 (FIXED 2026-08-21) BUG-06's notify fix didn't fully solve Baz's real-world "Telegram messages when tests run" complaint — a separate, systemic test-hygiene gap remained: several tests called real production notification code paths without mocking `send_telegram` at all. `test_gpu_recover.py`'s `test_recovery_invoked_with_correct_resume_cmd`/`test_allow_reboot_env_passed_through`/`test_recovery_failure_raises_runtime_error` called the real `kairos_gpu.ensure_cuda()` (whose GPU-recovery notifications are *intentionally* ungated by any enable flag — that's by design, not a bug); most of `test_kairos_papertrade.py`'s `TestPrewarmPredictionCache` tests called the real `prewarm_prediction_cache()` with `notify` left at its default `True`. Both silently no-op'd in a sandboxed/CI env (missing credentials → `OpsError`, logged as a warning) but fired for real on a dev machine with `TELEGRAM_BOT_TOKEN` already in the shell. Fixed with a global `autouse` fixture in `tests/conftest.py` that no-ops `send_telegram`/`send_telegram_document` in every module that imports either by name (`kairos_gpu`, `kairos_papertrade`, `kairos_pipeline`, `kairos_daily_signals`, `kairos_weekly_discovery`) for every test by default, rather than patching each individual test — a test that wants to verify real notify call content can still locally override it. Full suite green (1687 passed; the only 4 failures were a confirmed, unrelated collision with a live finetune batch job holding the real GPU lock at the same time, not a regression).
- [x] BUG-10 (FIXED 2026-08-21, commit `b4be6f9`) `calendar_days_for_bars` (`strategy/kairos_strategies.py`) corrected for weekends (equities/FX/futures trade ~5/7 days) but never for equities also trading only ~6.5 hours/day (NYSE), not 24 — `BARS_PER_DAY["1h"]=24` is crypto-calibrated, and FX/futures trade near-continuously on weekdays so 24/day approximates them fine, but it undersized equities' calendar-day window by ~24/6.5x. Found while tracing why zero equity signals (CRM, ABBV, CB, AAPL) appeared in a 2-week 1h signal-generation sweep despite no visible exception — the actual failure (`"Not enough data for CB: need 300 bars, got 252"`) was being silently swallowed by `kairos_signals.run()`'s per-group exception handling into a `failures` list a `return_rows=False` caller never sees; only surfaced by reading a written report file's own `## Failures` section. Fixed by adding `is_limited_hours_equity_symbol()` and rescaling `bars_per_day` by `EQUITY_TRADING_HOURS_PER_DAY=6.5` for plain-ticker equities specifically, only at intraday granularity (daily-interval equities unaffected). This same code path is also used by the oracle/base/finetuned backtest stages, so it likely also explains why only 8 of the 46 equities with 1h oracle data ever got a matching base run — re-running the discovery sweep for equities should now recover meaningfully more coverage. Live-verified: all 8 previously-failing equities now fetch 476 rows (vs 300 needed) at the exact timestamp that failed before. Full suite green (1696 passed).

## Epic 12 — Oracle stage for 1h + OrchestratorConfig calibration

- [x] E12-S01 Interval-keyed OrchestratorConfig preset mechanism (docs/tickets/E12-S01-orchestrator-config-interval-presets.md)
- [x] E12-S02 ⚠️ Live debug_filters=True calibration sweep for 1h (docs/tickets/E12-S02-orchestrator-1h-calibration-sweep.md) — n=4579 samples, 1h thresholds match 1d exactly (data-verified, not copied blindly); sample was thin/commodity-skewed at the time, now that BUG-05's $vol=0.0 fix has landed a broader crypto sample is available and re-running this sweep is worth considering
- [x] E12-S03 ⚠️ Live-verify oracle stage for 1h + playbook (docs/tickets/E12-S03-oracle-1h-live-verify-and-playbook.md) — run_id=737, group_id=5 (ZW=F), 127 strategies evaluated, 25 disabled for negative shadow Sharpe; disabled_strategies mechanism confirmed live for 1h

## Epic 13 — Base model backtest for 1h

- [x] E13-S01 ⚠️ Live-verify base stage for 1h + playbook (docs/tickets/E13-S01-base-model-1h-live-verify-and-playbook.md) — run_id=738, real GPU inference confirmed, 1d untouched (5592 rows before/after)

## Epic 14 — Finetuning loop for 1h

- [x] E14-S01 ⚠️ Live-verify finetune_next for 1h + playbook (docs/tickets/E14-S01-finetune-next-1h-live-verify-and-playbook.md) — registry id=177, ACCEPTED (base sharpe 7.81 -> ft 18.06), verified against DB+filesystem; found a real bug: select_finetune_candidate's already_registered check is interval-blind (no `interval` filter), permanently hiding candidates whose assets already have a 1d registry row of any status — see docs/playbooks/hourly-finetuning.md's "Known bug" section; fixed by E14-S02
- [x] E14-S02 Fix select_finetune_candidate's interval-blind already_registered check (docs/tickets/E14-S02-finetune-candidate-interval-blind-registry-check.md)

## Epic 15 — Signal generation + selection/allocation for 1h

- [x] E15-S01 ⚠️ Live-verify kairos_signals.py for 1h + update hourly-signals playbook (docs/tickets/E15-S01-signals-1h-live-verify-and-playbook.md) — 3 real runs across an hour boundary: cache hit within-hour (byte-identical, 44 rows), cache miss across boundary (88 rows, fresh content); Skipped/Failures footers legitimate, no bugs found

## Epic 16 — Papertrade/MTM/margin for 1h

- [x] E16-S01 Once-per-calendar-day financing/MTM guard (docs/tickets/E16-S01-papertrade-once-per-day-financing-guard.md)
- [x] E16-S02 ⚠️ Live-verify papertrade for 1h + playbook (docs/tickets/E16-S02-papertrade-1h-live-verify-and-playbook.md) — 2 runs, guard proven (1 kairos_mtm_daily row per calendar date across 3-4 date windows, not per hourly iteration); financing stayed 0 in both runs (no positions survived to day-close in this thin window) so the nonzero-accrual spot-check couldn't be positively exercised, see playbook for the structural argument covering that gap

## Epic 17 — Hourly Telegram digest + scheduling

- [ ] E17-S01 New systemd timer for hourly digest (docs/tickets/E17-S01-hourly-digest-systemd-timer.md)
- [ ] E17-S02 ⚠️ Live-verify digest + hand off timer-enable decision (docs/tickets/E17-S02-hourly-digest-live-verify-and-playbook.md)

---

See `docs/tickets/APPENDIX-A-standards.md` for code style, test conventions, and commit rules that apply to every story.
