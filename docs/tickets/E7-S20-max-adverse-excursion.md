# E7-S20 — Max adverse excursion (per-signal isolated drawdown) pure function

**Goal:** A pure function computing how far a position moved against itself, bar by bar, during its own isolated lifetime (not a portfolio-level drawdown).

**Context:**
- Read `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §3.3's `max_drawdown_pct` paragraph — this is new code, not something `BacktestEngine` already exposes.
- Read `strategy/kairos_mtm.py`'s `unrealized_pnl(pos, close_price)` (~line 75) for the exact direction-aware PnL convention already established in this codebase: long → `(price - entry) * qty`; short → `(entry - price) * qty`. Mirror this direction convention exactly (adapted to a % move instead of an absolute PnL) — do not invent a different sign convention.
- Read `strategy/kairos_backtest.py`'s `BacktestEngine._check_exit` (~line 1993) for the exact bar-field access pattern used elsewhere in this codebase (`float(bar["high"])`, `float(bar["low"])`, etc., lowercase column names) — confirm current column naming (`Open/High/Low/Close` vs `open/high/low/close`) against actual `price_cache.get_price_data` output before assuming casing, since it may differ from `_check_exit`'s internal `df` convention.

**Acceptance criteria:**
- [ ] `max_adverse_excursion_pct(direction: str, entry_price: float, bars: pd.DataFrame) -> float`: walks each row in `bars` (a DataFrame with OHLC columns), and for each bar computes the worst (most adverse) direction-aware % move using that bar's adverse-side extreme — for a long, the bar's Low; for a short, the bar's High — relative to `entry_price`. Tracks and returns the WORST (most negative economically, but returned as a POSITIVE percentage, e.g. `5.2` for a 5.2% adverse move) value seen across all bars. Returns `0.0` if the position was never underwater at any bar (or if `bars` is empty).
- [ ] Does not raise on an empty `bars` DataFrame — returns `0.0`.
- [ ] Unit tests hand-compute the expected value for a small synthetic bars DataFrame (3-4 rows with known Low/High values) for BOTH `direction="long"` and `direction="short"`, asserting exact match (same discipline as this session's `TestCorrectedCashFillCloseDelta` worked-example tests) — include one case where the position never goes underwater (expect `0.0`) and one where it does.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped).
- [ ] Max-adverse-excursion tests pass.
- [ ] Changes committed and `docs/todo.md` E7-S20 item checked off.
