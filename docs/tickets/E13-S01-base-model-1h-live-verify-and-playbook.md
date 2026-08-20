# E13-S01 — Live-run `--stage base --interval 1h`; write the playbook

**Goal:** Confirm the base-model backtest stage runs cleanly for `1h`; per the design doc this stage needs no code changes (`run_backtest_subprocess`/`refresh_disabled_strategies` already forward/key by interval correctly) — this story is verification + documentation only.

**⚠️ Requires GPU (or `KAIROS_ALLOW_CPU=1`) and real model inference. Execute manually or under supervision, not via unattended automation — a base-model backtest over enough bars can run for a while.**

**Context:**
- Depends on E12 (oracle-viable `1h` groups exist) — in practice `--stage base` can run for any group with enough price history, but running it against an oracle-viable group is the realistic pipeline order.
- Command: `uv run ./strategy/kairos_pipeline.py --stage base --interval 1h --group_id <id> --backtest_period <period>` (check `--help` for current flags).
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §2 (E4/E13 section).

**Acceptance criteria:**
- [ ] Run the base stage for `1h` against at least one real group. Command must exit 0.
- [ ] Inspect `model_results` (`SELECT * FROM model_results WHERE stage='base' AND interval='1h' ORDER BY run_id DESC LIMIT 20`) — plausible `sharpe`/`avg_pnl_per_trade`/`signal_count` values, same sanity bar as E12-S03.
- [ ] Confirm no `1d` `model_results` rows were touched by this run.
- [ ] Write `docs/playbooks/hourly-base-model.md` (new file, mirror `docs/playbooks/hourly-signals.md`'s structure): prerequisites, command, what success looks like, and a note that this stage's results feed the `finetune_next` comparison baseline (E14).

**Definition of done:**
- [ ] Live run completed, results reviewed for plausibility.
- [ ] `docs/playbooks/hourly-base-model.md` exists and is accurate.
- [ ] `docs/todo.md` E13-S01 item checked off.
