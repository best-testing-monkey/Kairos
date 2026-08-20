# E16-S02 — Live-run `kairos_papertrade.py --interval 1h`; write the playbook

**Goal:** Confirm the papertrade loop runs cleanly for `1h` with the E16-S01 financing guard in place, producing sane equity/MTM output over a short real window.

**⚠️ Requires GPU and real model inference across many iterations (24x more iterations/day than `1d` for the same wall-clock window). Run over a SHORT window first (a few days, not `--months-back 6`) — execute manually or under supervision; do not queue an open-ended run into unattended automation without first confirming a short run behaves.**

**Context:**
- Depends on E15 (signals for `1h` work) and E16-S01 (financing guard) being done.
- Command: `uv run ./strategy/kairos_papertrade.py --interval 1h --months-back <small, e.g. 0.25> ...` (check `--help` for current required flags — mirror whatever a normal `1d` papertrade invocation needs, per `docs/playbooks/` if a papertrade playbook already exists).
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §2 (E7/E16 section) for the financing-cadence reasoning this run is meant to validate.

**Acceptance criteria:**
- [ ] Run a short `--interval 1h` papertrade window. Command must exit 0 (or complete its intended window without crashing/OOM — reference `strategy/memory_monitor_heap.py`'s safety net and `docs/tickets/*prewarm*`/CLAUDE.md's leak-hunting history if it does crash, this codebase has been through several rounds of exactly this class of bug).
- [ ] Inspect `kairos_mtm_daily` rows written during the run (`SELECT date, COUNT(*) FROM kairos_mtm_daily GROUP BY date` for the run's window) — confirm **exactly one row per calendar date**, not one row per hourly iteration (this is the direct live-conditions check of E16-S01's guard).
- [ ] Spot-check `financing_accrued_total` (or the per-day `financing_accrued_day` column) over the run's window against a hand-calculated expectation (`daily_financing`'s `/360`-annualized rate × days held × position notional) — confirm it's NOT ~24x larger than expected (the exact bug E16-S01 fixes).
- [ ] Confirm the watchdog/forensics logging (`data/papertrade_watchdog.log`, per CLAUDE.md's "Watchdog forensics" section) shows no unexpected slow-iteration or crash entries for this run.
- [ ] Write `docs/playbooks/hourly-papertrade.md` (new file, mirror `docs/playbooks/hourly-signals.md`'s structure): prerequisites, command (with a note to start with a SHORT `--months-back` for a first `1h` run given the 24x iteration multiplier), what success looks like (one MTM row per day, not per hour), and a pointer back to E16-S01's guard as the mechanism that makes this safe.

**Definition of done:**
- [ ] Live run completed and reviewed against all acceptance criteria above.
- [ ] `docs/playbooks/hourly-papertrade.md` exists and is accurate.
- [ ] `docs/todo.md` E16-S02 item checked off.
