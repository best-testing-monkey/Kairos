# E4-S08 — Add CLI flags and load margin config in main

**Goal:** Add `--margin-config`, `--max-leverage`, and `--margin-utilization` CLI flags to `strategy/kairos_papertrade.py` and load the margin config in `main()`.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.7.
- Read `strategy/kairos_papertrade.py` `_build_arg_parser()` (around line 1344) and `main()` (around line 1430).
- Read `strategy/kairos_margin.py` (output of E1-S02) for `load_margin_config`.
- Read `strategy/allocation.py` (output of E3-S07) for `AllocationConfig`.
- Modify `strategy/kairos_papertrade.py`.

**Acceptance criteria:**
- [ ] `_build_arg_parser()` adds three arguments:
  - `--margin-config PATH`, default `config/margin_ibkr.yaml`;
  - `--max-leverage FLOAT`, default `1.0`;
  - `--margin-utilization FLOAT`, default `0.8`.
- [ ] `main()` loads `MarginConfig` via `load_margin_config(args.margin_config)` once per run.
- [ ] `main()` builds `AllocationConfig` with `max_leverage=args.max_leverage` and `margin_utilization_cap=args.margin_utilization`.
- [ ] `--max-leverage 1.0` keeps the legacy cash path unchanged.
- [ ] `_format_start_message()` and `_format_start_sim_message()` include `max_leverage` and `margin_utilization` in the notification text.
- [ ] Unit test(s) in `tests/unit/test_kairos_papertrade.py` assert the parser accepts the new flags and exposes the correct defaults.

**Definition of done:**
- [ ] `flake8` and `mypy` pass.
- [ ] Parser unit tests pass.
- [ ] Changes committed and `docs/todo.md` E4-S08 item checked off.
