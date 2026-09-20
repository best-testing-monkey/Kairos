# E21-S01 — Fix prediction-usage dispatch: naive-mode lookahead leak + missing interval

**Goal:** Fix two related bugs at the `PREDICTION_USAGE_MODES` dispatch call site in
`_run_day()` (`strategy/kairos_orchestrator.py`, around line 1127-1151), found by code review
of the E18/E19/E20 rollout. One is a real lookahead bug reopening a fix this codebase already
made once; the other silently corrupts bar spacing on any non-daily interval.

**Context — read the full review findings below before touching anything, then read current
source, since line numbers may have shifted slightly since the review:**

**Bug 1 (critical, needs design judgment, not just a patch):** In naive-baseline mode,
`distribution_as_bar` re-appends an approximation of the just-withheld real bar into
`context_histories`, leaking it into `returns_window`/`realized_vol` for every strategy.
`OrchestratorConfig(naive_baseline=True, prediction_usage_mode="distribution_as_bar")`:
`_make_realized_predictions(naive=True)` withholds the real last bar and builds a near-certain
`dist` from it; the dispatch's `usage_fn(pred)` then builds a synthetic bar from that same
`dist`'s pct_50 values and appends it to `pred.history` *before* `context_histories` is
computed. Since `context_histories` picks post-transform `p.history` when `naive_baseline` is
True, any strategy reading `context['returns_window']`/`context['realized_vol']` (or `history`
directly) sees a near-exact copy of the bar it's supposed to be blind to — reopening the
2026-09-01 lookahead fix documented in this repo's CLAUDE.md ("Oracle vs. naive-baseline
modes") via a second, uncomposed code path. No test exercises this combination.

Read that CLAUDE.md section in full before deciding on a fix — this project has been bitten by
this exact bug shape twice already and treats it as load-bearing. The composition of
`naive_baseline=True` with `prediction_usage_mode="distribution_as_bar"` is not obviously
meaningful in the first place (naive mode exists specifically to prove a strategy needs no
future information; re-injecting an approximation of the withheld bar defeats that by
construction, regardless of which code path it travels through). Recommended fix, unless you
find a cleaner one you can justify: **reject the combination explicitly** — raise a clear,
early error (e.g. at `OrchestratorConfig` construction or at the top of `_run_day()`) when both
`naive_baseline=True` and a non-identity `prediction_usage_mode` are set, rather than silently
composing them incorrectly. Document your reasoning either way.

**Bug 2 (mechanical):** `usage_fn(pred)` is called with no interval argument, so
`distribution_as_bar`'s `interval="1d"` default is always used regardless of the orchestrator's
real bar interval. `OrchestratorConfig` has no interval field threaded to this call site. On
any 1h/4h backtest (in production per `_FILTER_PRESETS_BY_INTERVAL["1h"]`) with
`prediction_usage_mode="distribution_as_bar"`, `_build_synthetic_bar`
(`strategy/kairos_prediction_usage.py`) advances the timestamp by `timedelta(days=1)` instead
of one real bar-interval, corrupting any indicator (RSI/MACD/ATR) reading bar spacing. No test
dispatches this mode through `_run_day` with a non-1d interval. Fix: thread the real interval
(from `self.config`/`KairosSettings.interval`, verify which is actually in scope at that call
site) through to `usage_fn`. This likely means changing `PredictionUsageFn`'s type signature to
accept an interval parameter — update the `"last_real_bar"` identity entry to accept and ignore
it, and `distribution_as_bar` to actually use it instead of its default.

**Acceptance criteria:**
- [ ] Bug 1: the `naive_baseline=True` + non-identity `prediction_usage_mode` combination either
  raises a clear, early, documented error, or is fixed so `context_histories`/`returns_window`/
  `realized_vol` never see the re-injected approximation of the withheld bar — your choice,
  justified in a code comment.
- [ ] Unit test: the naive+distribution_as_bar combination is exercised and does NOT leak the
  withheld bar's values into context (or is proven to raise cleanly, whichever fix you chose).
- [ ] Bug 2: `usage_fn` dispatch receives the orchestrator's real configured interval, not a
  hardcoded default. Unit test: dispatch `distribution_as_bar` through `_run_day()`-equivalent
  code with a non-"1d" interval (e.g. "1h") and assert the synthetic bar's timestamp is exactly
  one real interval-step ahead, not one day.
- [ ] `PREDICTION_USAGE_MODES["last_real_bar"]` (identity mode) still works unchanged after any
  signature change — regression test.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `strategy/kairos_orchestrator.py` and
  `strategy/kairos_prediction_usage.py`.
- [ ] New tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`).
- [ ] Commit with "E21-S01" in the subject line. No Co-Authored-By trailer (see
  `docs/tickets/APPENDIX-A-standards.md`).
- [ ] Do NOT edit `docs/todo.md` — the orchestrating session checks stories off after verifying.
