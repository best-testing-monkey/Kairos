# E18-S05 — Offline comparison: `MLBracketStrategy` vs. static-percentile baseline

**Goal:** Reuse `kairos_signal_replay.py`'s replay machinery to compare a strategy wrapped in
`MLBracketStrategy` against its unwrapped (static-percentile) baseline over a held-out window,
producing an EV/Sharpe comparison report. This is the gate before live wiring is even considered
(per `DESIGN_DOC_ml_tpsl_optimization.md` §5's non-goals).

**Context:**
- Depends on E18-S04 (`MLBracketStrategy`).
- `strategy/kairos_signal_replay.py`'s `replay()` (~line 898) and `compute_closures_for_window()`
  (~line 757) are the existing offline comparison primitives — read them before writing anything
  new; this story should be a thin comparison wrapper around existing replay output, not a new
  backtest engine.
- The comparison needs two closure sets over the **same held-out window**: one for the base
  strategy's original signals (already in `papertrade_signals_closure` from a normal
  `--precompute` run), and one for the same signals with `MLBracketStrategy`-chosen stop/target
  substituted in. Since `MLBracketStrategy` only changes `.stop`/`.target` (not entry/direction),
  the comparison can reuse E18-S01's own per-candidate resolution helper (whichever function that
  story produced — extracted-shared or duplicated, per its "implementer's call" note) against the
  `MLBracketStrategy`-chosen candidate per signal, rather than needing a third resolution
  implementation.
- Held-out window: use whatever slice E18-S03's purged-CV validation split held out (read
  `scripts/train_tpsl_model.py`'s split logic to get the exact date range, don't pick a new one
  arbitrarily — mixing training and validation data into this comparison would invalidate it).

**Acceptance criteria:**
- [ ] New script or flag (implementer's call: extend `scripts/train_tpsl_model.py` with a
  `--validate` mode, or a new `scripts/compare_tpsl_model.py` — whichever is the smaller diff)
  produces a report: for each strategy in a configurable list, baseline EV/Sharpe/win-rate vs.
  `MLBracketStrategy`-wrapped EV/Sharpe/win-rate over the held-out window, same signal set.
- [ ] Report states explicitly (in output, not just this doc) that this is a **directional**
  comparison, not a live-P&L claim — mirrors `kairos_signal_replay.py`'s own documented cost-model
  caveat (`DESIGN_DOC_offline_signal_replay.md` §3.3).
- [ ] Unit test: synthetic fixture with a known baseline vs. ML-candidate outcome difference —
  report correctly attributes the EV delta to the right strategy/candidate.
- [ ] Uses the exact held-out window from E18-S03's split, not a separately-chosen one (verify by
  reading the split boundary from wherever E18-S03 persisted it — a config file, a logged value,
  or a constant; if E18-S03 didn't persist it anywhere machine-readable, that's a gap to flag back
  on that story rather than silently guessing a new window here).

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to the new/changed script.
- [ ] New tests pass; full suite green.
- [ ] Changes committed and `docs/todo.md` E18-S05 item checked off.
