# E15-S01 — Live-run `kairos_signals.py --intervals 1h`; write the playbook

**Goal:** Confirm signal generation + selection/allocation works end-to-end for `1h` against real oracle/base/finetuned results, exercising the E0 `_cache_as_of_value` fix from this same design doc's already-shipped slice under real conditions (not just the unit-test mocks).

**Requires GPU (or `KAIROS_ALLOW_CPU=1`) for model inference. Can be run semi-attended — this is a single `kairos_signals.py` invocation, not an open-ended training loop — but still budget real GPU time and review the output report by hand at least once.**

**Context:**
- Depends on E12/E13/E14 (oracle/base/finetuned `1h` results should exist so signal generation has real strategies/models to work with — though `kairos_signals.py` will run even with a thin `1h` `viability_report`, just producing fewer/no signals).
- Note: `docs/playbooks/hourly-signals.md` **already exists and already documents this exact command** (`uv run ./strategy/kairos_signals.py --intervals 1h --xlsx`) — this story is about actually RUNNING it for the first time against a fully-populated `1h` pipeline (universe→correlation→oracle→base→finetuned all done) and confirming the output, then updating that existing playbook with anything learned, not writing a new one from scratch.
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §2 (E6/E15 section) and the existing `docs/playbooks/hourly-signals.md` in full before starting.

**Acceptance criteria:**
- [ ] Run `uv run ./strategy/kairos_signals.py --intervals 1h --xlsx` (or `--gsheets`/`--ods`, whichever output format is convenient) against a fully-populated `1h` pipeline. Command must exit 0.
- [ ] Inspect the generated report: confirm the `## Skipped` footer (if present) doesn't show every single strategy skipped as "unknown strategy" (a full-skip result would indicate `resolve_disabled_strategies`/strategy registry issues specific to `1h`, worth flagging).
- [ ] Confirm `signals_cache` rows were written with `interval='1h'` (`SELECT COUNT(*) FROM signals_cache WHERE interval='1h'`).
- [ ] Run the SAME command a second time within the same clock hour; confirm (via timing, or by adding a temporary print/log if needed) that the second run hits the `signals_cache` (fast) rather than recomputing — this is the direct live-conditions check of the E0 `_cache_as_of_value` fix.
- [ ] Wait until the next hour boundary and run again; confirm this THIRD run does NOT reuse the first run's cached rows (i.e. produces fresh data) — the other half of the same fix.
- [ ] Update `docs/playbooks/hourly-signals.md`'s "Hourly-specific caveats" section with anything learned from this live run that isn't already documented there (e.g. real EV-floor binding behavior, real disabled-strategy-set differences observed vs. assumed).

**Definition of done:**
- [ ] Live run(s) completed and reviewed.
- [ ] `docs/playbooks/hourly-signals.md` updated (not replaced) with real findings.
- [ ] `docs/todo.md` E15-S01 item checked off.
