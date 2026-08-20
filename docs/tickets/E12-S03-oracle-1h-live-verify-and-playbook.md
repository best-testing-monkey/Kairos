# E12-S03 — Live-run `--stage oracle --interval 1h`; write the playbook

**Goal:** Confirm the oracle stage runs cleanly end-to-end for `1h` using the calibrated presets from E12-S02, then document the flow.

**⚠️ Requires GPU (or `KAIROS_ALLOW_CPU=1`) and real model inference — same caution as E12-S02. Execute manually or under supervision, not via unattended automation.**

**Context:**
- Depends on E12-S01, E12-S02, and a `1h` `suggested_groups` entry from E11.
- Command: `uv run ./strategy/kairos_pipeline.py --stage oracle --interval 1h --group_id <id> --backtest_period <period>` (use a `group_id` from the live E11 correlation run; check `--help` for the current default `--backtest_period`).
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §2 (E3/E12 section) — this story is the "run it and see" half of E3/E12, oracle itself needs no code changes (it already forwards `--interval` cleanly per the design doc).

**Acceptance criteria:**
- [ ] Run the oracle stage for `1h` against at least one real group. Command must exit 0.
- [ ] Inspect `oracle_results` (`SELECT * FROM oracle_results WHERE interval='1h' ORDER BY run_id DESC LIMIT 20`) — rows should have plausible `sharpe`/`avg_pnl_per_trade`/`signal_count` values (not all-zero, not NaN, not absurdly large — the `_safe_sharpe` clamp from `kairos_orchestrator.py` bounds pathological cases, so a hugely out-of-range value here would indicate a real bug, not just noisy data).
- [ ] Confirm `refresh_disabled_strategies`/`disabled_strategies` picked up a `1h` profile if the oracle run's results warranted disabling any strategy (`SELECT * FROM disabled_strategies WHERE interval='1h'`) — an empty result is fine (means nothing warranted disabling yet), just confirm the query runs and the mechanism is live for `1h`.
- [ ] Write `docs/playbooks/hourly-oracle.md` (new file, mirror `docs/playbooks/hourly-signals.md`'s structure): prerequisites (universe + correlation for 1h, filter presets calibrated via E12-S02), the command, what a successful run's `oracle_results` should look like, and a note that oracle results feed `--stage base`/`--stage finetuned` (E13/E14) next.

**Definition of done:**
- [ ] Live run completed, results reviewed for plausibility (not just "exit code 0").
- [ ] `docs/playbooks/hourly-oracle.md` exists and is accurate to what was actually observed.
- [ ] `docs/todo.md` E12-S03 item checked off.
