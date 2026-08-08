# E2-S06 — Add financing and borrow-cost accrual

**Goal:** Add `daily_financing` to `strategy/kairos_mtm.py` to compute per-position overnight financing/borrow costs.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.5.
- Read `strategy/kairos_mtm.py` (output of E2-S03/E2-S05) and `strategy/kairos_margin.py` (output of E1-S02).
- Modify `strategy/kairos_mtm.py`.
- Add tests to `tests/unit/test_kairos_mtm.py`.

**Acceptance criteria:**
- [ ] `daily_financing(pos: OpenPositionView, close_price: float, cls: MarginClass, cfg: MarginConfig) -> float` is defined.
- [ ] Spot classes (`initial_margin_pct == 100.0`) return `0.0`.
- [ ] Long CFD/margin: `notional_close * (benchmark_annual_pct + financing_spread_pct) / 360`.
- [ ] Short CFD/margin: credit `(benchmark_annual_pct - financing_spread_pct) / 360` if positive, plus always debit `notional_close * short_borrow_annual_pct / 360`.
- [ ] `notional_close = close_price * pos.quantity`.
- [ ] A helper `compute_daily_financing_total(positions, bars_by_ticker, cfg) -> float` sums per-position financing for a day.
- [ ] Unit tests cover:
  - long financing exact value,
  - short with borrow exact value,
  - spot returns 0,
  - benchmark/spread zero edge cases.
- [ ] Docstring documents the convention: financing charged on positions open at bar close; entry day counts, exit day does not.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] Financing unit tests pass.
- [ ] Changes committed and `docs/todo.md` E2-S06 item checked off.
