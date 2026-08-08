# E2-S05 — Add liquidation check

**Goal:** Add `liquidation_check` to `strategy/kairos_mtm.py` that determines which positions to liquidate under the ESMA 50% close-out rule.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.4.
- Read `strategy/kairos_mtm.py` (output of E2-S03/E2-S04) and `strategy/kairos_margin.py` (output of E1-S02).
- Modify `strategy/kairos_mtm.py`.
- Add tests to `tests/unit/test_kairos_mtm.py`.

**Acceptance criteria:**
- [ ] `liquidation_check(snapshot: DailySnapshot, positions: list[OpenPositionView], cfg: MarginConfig) -> tuple[list[str], float, bool]` returns `(tickers_liquidated, post_equity, ruined)`.
- [ ] Trigger condition: `snapshot.equity < cfg.closeout_fraction * snapshot.initial_margin_used`.
- [ ] When not triggered, returns `([], snapshot.equity, False)`.
- [ ] When triggered, repeatedly removes the position that releases the largest maintenance margin (`notional * maintenance_margin_pct`) first.
- [ ] After each simulated removal, recomputes `equity_post` and `initial_margin_used_post`.
- [ ] Stops when `equity_post >= cfg.closeout_fraction * initial_margin_used_post` or no positions remain.
- [ ] If all positions are liquidated and equity would still be negative, clamps `equity_post` to `0.0` and returns `ruined=True`.
- [ ] Unit tests cover:
  - no trigger when equity >= 50% of IM,
  - trigger when equity = 49% of IM,
  - deterministic ordering (largest MM release first),
  - post-liquidation invariant holds,
  - clamp-to-zero + ruined when equity cannot be restored.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] Liquidation unit tests pass.
- [ ] Changes committed and `docs/todo.md` E2-S05 item checked off.
