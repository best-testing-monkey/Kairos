# E11-S02 — Live-run `--stage correlation --interval 1h`; write the playbook

**Goal:** Confirm the correlation stage produces sane, real `suggested_groups` for `1h` against live price data (not just mocked unit tests), then document the flow.

**Context:**
- Depends on E10 (universe-for-1h) and E11-S01 (interval-scaled windows) being done and committed. Requires the E10 universe stage to have actually been run for `1h` first (`uv run ./strategy/kairos_pipeline.py --stage universe --interval 1h`) so `universe_screen` has `1h` survivor rows to read.
- This is a **live-data verification story, not a code-change story** — no GPU needed (correlation is pure price-data math), but it does make real network/`price_cache` calls and can take a few minutes. Safe to run unattended.
- Command: `uv run ./strategy/kairos_pipeline.py --stage correlation --interval 1h`
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §2 (E2/E11 section) for context on what "success" means here.

**Acceptance criteria:**
- [ ] Run `uv run ./strategy/kairos_pipeline.py --stage universe --interval 1h` first if `data/pipeline_results.db`'s `universe_screen` table has no `1h`-interval rows yet (check with `sqlite3 data/pipeline_results.db "SELECT COUNT(*) FROM runs WHERE stage='universe' AND interval='1h'"`).
- [ ] Run `uv run ./strategy/kairos_pipeline.py --stage correlation --interval 1h`. Command must exit 0.
- [ ] Inspect the printed summary and the `suggested_groups` table (`SELECT * FROM suggested_groups WHERE run_id = (SELECT MAX(run_id) FROM runs WHERE stage='correlation' AND interval='1h')`) — at least one group or singleton row should exist (a completely empty result for every survivor would indicate the interval-scaled `min_overlap` from E11-S01 is too strict for real 1h data availability — if that happens, note it as a finding rather than silently treating the run as successful).
- [ ] Confirm no `1d` data was touched: `SELECT COUNT(*) FROM runs WHERE stage='correlation' AND interval='1d'` should be unchanged from before this run.
- [ ] Write `docs/playbooks/hourly-correlation.md` (new file, mirror `docs/playbooks/hourly-signals.md`'s structure): prerequisites (1h universe stage must have run first), the command, what a successful run's output looks like, and a caveat that `min_overlap`/`roll_window` are scaled 24x vs `1d` (from E11-S01) so a thin 1h price history (yfinance's 729-day cap) may disqualify more pairs than the `1d` run does for the same assets.

**Definition of done:**
- [ ] Live run completed successfully (or findings documented if it wasn't fully clean — don't force a pass).
- [ ] `docs/playbooks/hourly-correlation.md` exists and is accurate to what was actually observed.
- [ ] `docs/todo.md` E11-S02 item checked off (no code changes to commit unless the playbook doc itself counts — commit the playbook doc).
