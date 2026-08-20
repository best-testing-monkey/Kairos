# E14-S01 — Live-run `--stage finetune_next --interval 1h`; write the playbook

**Goal:** Confirm the automated finetuning loop runs cleanly for `1h`. Per the design doc, `finetuned_models`'s registry is already interval-safe (`UNIQUE(assets, interval)`, checkpoint dirs named `{interval}__{assets}/`) and `compute_finetune_periods`/`_YF_MAX_DAYS` already correctly cap `1h` training history at 729 days — this story is verification only, no code changes expected.

**⚠️ Requires GPU and a real (potentially long) model training run. Do NOT queue this into unattended `/run-stories` automation — finetune training can run for a substantial amount of wall-clock time and needs a human decision on whether to accept the result. Execute manually or under direct supervision, and budget real time for it.**

**Context:**
- Depends on E13 (a `1h` base-model comparison baseline should exist, since `finetune_next` compares finetuned-vs-base viability).
- Command: `uv run ./strategy/kairos_pipeline.py --stage finetune_next --interval 1h` (check `--help` — this stage typically operates DB-wide rather than taking `--group_id`, per `strategy/kairos_pipeline.py`'s `select_finetune_candidate`/`compare_finetuned_vs_base`, which group by `(assets, interval, backtest_period)`).
- Telegram notifications fire automatically for this stage (🟢 start, ❌ training failure, ✅/⚠️ accept/reject verdict, 💥 crash) per CLAUDE.md's "Telegram notifications" section — make sure `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` are sourced into the shell first (`set -a && source ~/.config/kairos/kairos.env && set +a`) or notifications will silently no-op with only a `WARNING:` log line.
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §2 (E5/E14 section).

**Acceptance criteria:**
- [ ] Run `--stage finetune_next --interval 1h`. Confirm the printed `[finetune_next] periods:` line shows a `train_start` no further back than 729 days before `train_end`/`test_start` (yfinance's 1h history cap) — this is `compute_finetune_periods`'s `_YF_MAX_DAYS["1h"] = 729` already in effect; if the printed window is longer than 729 days, that's a real bug to report, not something to patch silently in this ticket.
- [ ] Confirm a `models/finetuned/1h__<assets>/` checkpoint directory is created (matching `finetune_model_dir`'s naming convention).
- [ ] Confirm the `finetuned_models` registry row has `interval='1h'` and the checkpoint path matches (`SELECT * FROM finetuned_models WHERE interval='1h' ORDER BY id DESC LIMIT 5`).
- [ ] Whatever the verdict (accepted/rejected), confirm the printed `[finetune_next] VERDICT:` line and the registry `status` column agree.
- [ ] Write `docs/playbooks/hourly-finetuning.md` (new file, mirror `docs/playbooks/hourly-signals.md`'s structure and cross-reference `docs/playbooks/model-finetuning.md`'s existing "Notifications" section rather than repeating it): prerequisites, command, expected training-window caveat (729-day cap), and how long to expect this to take relative to the `1d` finetune loop.

**Definition of done:**
- [ ] Live run completed (accept or reject is both an acceptable outcome — the point is confirming the machinery works, not forcing acceptance).
- [ ] `docs/playbooks/hourly-finetuning.md` exists and is accurate.
- [ ] `docs/todo.md` E14-S01 item checked off.
