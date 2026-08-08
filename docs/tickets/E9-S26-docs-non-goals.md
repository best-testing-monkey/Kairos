# E9-S26 — Document non-goals in --help and module docstring

**Goal:** Make the tool's key limitations (unleveraged-only, cost-model divergence from live execution, single-interval-per-replay) visible to anyone running it, not just readable in the design doc.

**Context:**
- Read `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §4 (Non-goals) in full.
- Read `strategy/kairos_signal_replay.py` (own module — all prior outputs, including E9-S24's `_build_arg_parser()`). This story adds ONLY documentation — no new logic, no behavior changes.

**Acceptance criteria:**
- [ ] Module docstring at the top of `strategy/kairos_signal_replay.py` states, concisely (a short paragraph, not an essay): (1) this tool is unleveraged-only — no margin, CFD, or liquidation simulation; (2) its cost model (flat fee/slippage via `BacktestEngine`) diverges from `phantom`'s live per-instrument cost model, so results are directional signals, not P&L predictions; (3) any promising selection/allocation rule found here must still be validated with a real `kairos_papertrade.py` run before being trusted.
- [ ] `--help` text (via argparse `description=` on the parser, and/or per-flag `help=` strings on `--precompute`/`--replay`) reflects the same three points concisely enough that a user running `--help` sees them without needing to read the design doc.
- [ ] Verify via `git diff strategy/kairos_signal_replay.py` before committing that ONLY docstrings/help strings changed — no logic, no test behavior changes in this story.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped) — trivial since no logic changed.
- [ ] `git diff` confirms docs-only change.
- [ ] Changes committed and `docs/todo.md` E9-S26 item checked off.
