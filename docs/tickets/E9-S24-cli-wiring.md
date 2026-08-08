# E9-S24 — CLI wiring (--precompute / --replay)

**Goal:** Make `strategy/kairos_signal_replay.py` directly runnable via `uv run`, with `--precompute` and `--replay` entrypoints.

**Context:**
- Read `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §3.5.
- Read `strategy/kairos_papertrade.py`'s `_build_arg_parser()` (~line 1346) and its `if __name__ == "__main__":` tail for this repo's argparse conventions (help text style, `dest=` naming, default values) — mirror this style, don't invent a different one.
- Read `strategy/signal_selection.py`'s `parse_signal_selection`/`SignalSelectionRule`/`SignalSelectionError` and how `strategy/kairos_papertrade.py`'s `main()` already uses them (`--signal-selection` flag, `try: parsed = parse_signal_selection(args.signal_selection) except SignalSelectionError: parser.error(...)`) — reuse this exact pattern, do not reimplement selection-rule parsing.
- Read `strategy/kairos_signal_replay.py` (own module — outputs of E6-S18 `unpack_signals_cache_to_papertrade_signals`, E7-S21 `compute_closures_for_window`, E8-S23 `replay`).
- Read `strategy/kairos_signals.py`'s `DB_PATH` constant for the `--db` flag's default.

**Acceptance criteria:**
- [ ] `_build_arg_parser()`/`main(argv=None)` added to `strategy/kairos_signal_replay.py`, with an `if __name__ == "__main__": main()` (or `sys.exit(main())`, match `kairos_papertrade.py`'s exact convention) tail.
- [ ] `--precompute` flag: when set, calls `unpack_signals_cache_to_papertrade_signals` then `compute_closures_for_window` over a window derived from `--start`/`--end` (or `--months-back`, implementer's call — document which was chosen and why in a code comment), using `--interval-ladder` (comma-separated string, e.g. `"1h,4h,1d"`, split and passed as the ladder list) and `--db` (default = `kairos_signals.DB_PATH`).
- [ ] `--replay` flag: when set, calls `replay(...)` with `--interval` (single value, required for this mode), `--start`, `--end`, `--capital`, and `AllocationConfig`-mapped flags sufficient to be useful (at minimum `--max-pos-pct`, `--top-k`, `--signal-selection` — reusing `parse_signal_selection` exactly as `kairos_papertrade.py` does). No leverage/margin flags — this tool is unleveraged-only, do not add `--max-leverage` or similar.
- [ ] Prints the resulting metrics dict (JSON via `json.dumps`, matching `write_json_report`'s style, or pretty-printed — implementer's call, document choice).
- [ ] Unit tests (`tests/unit/test_kairos_signal_replay.py`): `_build_arg_parser()` accepts both `--precompute` and `--replay` flag combinations with correct defaults (same style as `test_kairos_papertrade.py`'s parser tests, e.g. `TestBuildArgParser`).

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped).
- [ ] Parser tests pass.
- [ ] Changes committed and `docs/todo.md` E9-S24 item checked off.
