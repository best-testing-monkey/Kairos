# E5-S15 — Leverage-off regression and exposure-cap verification

**Goal:** Verify `--max-leverage 1.0` reproduces legacy behavior and that the admission check bounds concurrent exposure.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §6.1 and §8.
- Read `docs/papertrade_tickets/02-portfolio-exposure-cap.md` for the exposure-cap motivation.
- Read `tests/unit/test_kairos_papertrade_loss_repro.py` for fixture patterns.
- Read `strategy/kairos_papertrade.py` `main()` and `AllocationConfig` (outputs of E4-S08, E4-S10).
- Create/modify `tests/unit/test_kairos_papertrade_leverage_regression.py`.

**Acceptance criteria:**
- [ ] A regression test runs `kairos_papertrade.main([...])` with `--max-leverage 1.0 --margin-utilization 0.8` on a short window and compares key metrics to a pre-recorded baseline; they match within tolerance.
- [ ] The baseline is stored in the test file as a pinned dict (same style as `EXPECTED_METRICS_V2`).
- [ ] A separate test asserts that with `--max-leverage 2.0 --margin-utilization 0.1` on a window known to generate many signals, peak `gross_notional` is bounded by `equity * max_leverage * margin_utilization_cap / min_initial_margin_pct`.
- [ ] Tests mock or skip GPU/network/phantom as needed; if phantom is required, use the fixture DB and skip when missing.
- [ ] Tests run with `uv run --with pytest python -m pytest tests/unit/test_kairos_papertrade_leverage_regression.py -q`.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] Regression tests pass (or skip gracefully).
- [ ] Changes committed and `docs/todo.md` E5-S15 item checked off.
