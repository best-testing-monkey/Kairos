# E20-S05 — Offline comparison: `distribution_as_bar` vs. `last_real_bar`

**Goal:** Reuse `kairos_signal_replay.py`-style tooling to compare indicator-gated strategies'
performance under `distribution_as_bar` vs. the default `last_real_bar` mode over a held-out
window — the gate before considering this mode for live use, per
`DESIGN_DOC_prediction_usage_mode_distribution_as_bar.md` §7's implementation order.

**Context:**
- Depends on E20-S04 (confirms which strategies are actually affected — scope this comparison to
  those: `RSIFilterStrategy`, `MACDFilterStrategy`, and any other `history`-reading strategy found
  along the way).
- Mirrors E18-S05's shape (same kind of offline replay comparison, different axis — prediction
  usage mode instead of stop/target model). Read that story's Context section for the
  `kairos_signal_replay.py` primitives to reuse (`replay()`, `compute_closures_for_window()`) —
  do not duplicate that research, just follow the same pattern.
- This comparison needs to re-run signal *generation* itself (not just resolve an already-decided
  signal's outcome, unlike E18-S05) — since `distribution_as_bar` changes what a strategy's
  `generate_signal()` call actually decides, not just how its stop/target gets resolved
  afterward. This means it needs `KairosOrchestrator.run_backtest()` (or `_run_day()` directly)
  with `OrchestratorConfig(prediction_usage_mode=...)` set to each mode in turn over the same
  historical window, not the existing `papertrade_signals`-replay path — flag this distinction in
  a code comment since it's easy to conflate with E18-S05's simpler resolve-only comparison.

**Acceptance criteria:**
- [ ] Script or test harness runs the same historical window twice — once per
  `prediction_usage_mode` — for the scoped set of indicator-gated strategies, and reports
  signal-count/EV/Sharpe deltas between the two modes.
- [ ] Report states explicitly that this compares signal-generation behavior, not just exit
  resolution — distinguishing it from E18-S05's comparison shape.
- [ ] Unit test: synthetic fixture with a crafted case where the synthetic bar flips an
  RSI/MACD-gated strategy's decision — report correctly shows a signal-count or EV difference
  between the two modes for that case.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to the new/changed script.
- [ ] New tests pass; full suite green.
- [ ] Changes committed and `docs/todo.md` E20-S05 item checked off.
