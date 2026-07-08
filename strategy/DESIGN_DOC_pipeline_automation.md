# Kairos Framework: Pipeline Automation Design Document (stages 1→4 + viability report)

**Status:** DESIGN — nothing in this document is implemented yet.
**Scope:** `strategy/kairos_pipeline.py`, `strategy/PIPELINE.md`, new unit tests. No changes to strategy logic, the orchestrator, or persistence schemas beyond what is specified here.

## 1. Overview

Today the asset-discovery pipeline (see `strategy/PIPELINE.md`) requires one manual
invocation per stage, manual asset selection between stages 2→3, and offers no
consolidated output. This document specifies:

1. A new `--stage auto` that chains **universe → correlation → oracle → base**
   (stages 1–4) in one command, iterating all suggested correlation groups and
   one or more bar intervals.
2. A **viability report**: a CSV (and DataFrame) of all strategies with
   instrument(s), bar interval, signals per week, oracle/base Sharpe, and a
   `viable` flag.

Decisions already made (do not re-litigate during implementation):
- **Viable** = `oracle_sharpe > 0 AND base_sharpe > 0 AND min(oracle_signals, base_signals) >= 3`,
  thresholds overridable via `--min_sharpe` (default 0.0) and `--min_signals` (default 3).
- **Asset scope** = all rows of `suggested_groups` from the stage-2 run
  (optionally filtered by `--asset_class`).
- **Intervals** = `--intervals 1d 1h ...` runs the full chain per interval.
- Stage 5 (finetuned) is explicitly **out of scope**; the report join must be
  written so a later `stage="finetuned"` column can be added without schema change.

## 2. New stage: `auto`

### 2.1 CLI

```
uv run ./strategy/kairos_pipeline.py --stage auto \
    [--intervals 1d [1h ...]] [--backtest_period 6m] [--asset_class crypto] \
    [--pred_samples 100] [--min_sharpe 0.0] [--min_signals 3] \
    [--force] [--skip_universe] [--report_only]
```

- `--intervals`: nargs="+", default `["1d"]`. The existing singular `--interval`
  remains for the individual stages and is rejected (argparse error) in
  combination with `--stage auto`.
- `--backtest_period`: passed through to stages 3–4 (already supported).
- `--force`: re-run (group, interval, stage) combos even if results exist.
- `--skip_universe`: start from the latest existing universe/correlation runs
  for each interval (useful when re-running after a crash without re-screening).
- `--report_only`: skip all execution; rebuild the viability report from the
  latest matching DB rows.

### 2.2 `run_stage_auto(conn, intervals, backtest_period, asset_class_filter, pred_samples, min_sharpe, min_signals, force, skip_universe) -> pd.DataFrame`

Per interval, in order:

1. **Universe:** `run_stage_universe(conn, interval)` (already interval-aware),
   unless `--skip_universe` and a prior universe run exists for this interval.
2. **Correlation:** `run_stage_correlation(conn, asset_class_filter)`.
   ⚠ Implementation item: verify the correlation stage's price fetch honors the
   interval (it currently assumes 1d closes). If it does not, thread `interval`
   through it — correlation on 1h bars must use 1h closes over an equivalent
   calendar window. Reuse `calendar_days_for_bars` from `kairos_strategies.py`
   for the window math; do not invent new bar/calendar logic.
3. **Groups:** read `suggested_groups` for the correlation `run_id` just created
   (or the latest one under `--skip_universe`). Each row's symbol list becomes
   one (assets, interval) work item.
4. **Per group:** `run_stage_oracle(conn, assets, interval, backtest_period,
   pred_samples)` then `run_stage_model(conn, "base", assets, interval,
   backtest_period, pred_samples, model_path=None)` — the existing functions and
   `run_backtest_subprocess` are reused **unchanged** (they already print the
   built/disabled/evaluating/fired counts and persist to
   `oracle_results` / `model_results` with a fresh `run_id`).
5. **Report:** after all intervals/groups, call `build_viability_report(...)`
   (§3), write the CSV, print the summary.

**Resumability:** before each stage-3/4 execution, query the target table for an
existing row set matching `(assets_key, interval, backtest_period)` (and
`stage="base"` for `model_results`); if present and not `--force`, log
`[skip] oracle BTC-USD,ETH-USD @1d (run_id=NNN exists)` and continue. Assets key
is the same comma-joined sorted string the tables already store.

**Failure isolation:** wrap each group's stage-3/4 call in try/except;
`RuntimeError` from `run_backtest_subprocess` (nonzero exit) logs the group and
continues with the next work item. Track `failures: List[dict]` and print a
summary block at the end; exit code 0 if at least one group completed, 1 if all
failed. Never let one bad ticker group kill an overnight run.

**Run bookkeeping:** insert one `runs` row with `stage="auto"` and
`params_json` = full parameter dict, so a report can later be tied to the exact
invocation.

## 3. Viability report

### 3.1 `build_viability_report(conn, intervals, backtest_period, min_sharpe=0.0, min_signals=3, asset_class_filter=None) -> pd.DataFrame`

Module-level function in `kairos_pipeline.py` (usable standalone via
`--report_only`).

- **Join:** for each interval, take the **latest** `oracle_results` and
  `model_results` rows with `stage="base"` per
  `(strategy_name, assets, interval, backtest_period)` (latest = max `run_id`
  for that key — re-runs supersede older rows), inner-joined on that key.
  Strategies present in only one side appear with the other side's columns NaN
  and `viable=False` (use an outer join, then gate) — the report must show
  *all* strategies that fired anywhere, per the user requirement.
- **Columns (exact, in order):**
  `strategy_name, assets, asset_class, interval, backtest_period,
  oracle_sharpe, base_sharpe, oracle_signals, base_signals, signals_per_week,
  win_rate, avg_pnl_per_trade, viable`.
  - `asset_class` via the existing `asset_class_for(assets)` helper in
    `kairos_strategies.py`.
  - `win_rate` / `avg_pnl_per_trade` from the base row (fall back to oracle row
    when base side is NaN).
- **signals_per_week** = `base_signals / weeks`, where `weeks` is derived from
  `backtest_period` with a shared helper `_period_to_weeks(period: str) -> float`
  implemented on top of the same parsing used by `_period_to_bars` in
  `kairos_strategies.py` (`"6m"` → 26.1, `"1m"` → 4.35, `"2w"` → 2.0; use
  365.25/12 days per month, 7 per week). Reuse the existing period parser —
  do not write a second regex. When the base side is NaN, compute from
  `oracle_signals` instead and leave a `signals_source` note out of scope
  (single column only; base preferred).
- **viable** = `(oracle_sharpe > min_sharpe) & (base_sharpe > min_sharpe) &
  (minimum of the two signal counts >= min_signals)`; NaN on either side →
  `False`.
- **Sort:** viable rows first, then `base_sharpe` descending.

### 3.2 Output

- CSV via the existing `dump_csv` convention:
  `results/auto_viability_report_<YYYYmmdd_HHMMSS>.csv` (keep the
  `<stage>_<name>_<timestamp>` pattern).
- Also persist to a new SQLite table `viability_report`
  (`run_id, strategy_name, assets, asset_class, interval, backtest_period,
  oracle_sharpe, base_sharpe, oracle_signals, base_signals, signals_per_week,
  win_rate, avg_pnl_per_trade, viable`) so the DB remains the source of truth
  per PIPELINE.md convention. `CREATE TABLE IF NOT EXISTS` alongside the other
  schema statements.
- Print: `Viability report: N strategies, V viable (interval breakdown: 1d: x/y, 1h: ...)`
  plus the CSV path.

### 3.3 Interaction with disabled strategies

`resolve_disabled_strategies(interval, assets)` continues to gate what the
subprocess runs; disabled strategies never appear in the results tables and
therefore never appear in the report. Document in PIPELINE.md that the report
covers **enabled** strategies only, and that the printed
`built X, disabled Y, evaluating Z` line is where the excluded count is visible.

## 4. PIPELINE.md updates

- New "Stage auto" section: the one-command flow, flags, resumability/`--force`
  semantics, report location, and a worked example:
  `uv run ./strategy/kairos_pipeline.py --stage auto --intervals 1d --backtest_period 3m --asset_class crypto`.
- Note stage 5 (finetuned) remains manual and how a finetuned column could later
  extend the report.

## 5. Testing plan

All tests in new `tests/unit/test_pipeline_auto.py`, no GPU/model/network:

1. **Auto-stage chaining:** monkeypatch `run_backtest_subprocess` with a stub
   returning a canned payload; assert call order (universe → correlation →
   per-group oracle → per-group base), one pair of calls per suggested group,
   repeated per interval for `--intervals 1d 1h`.
2. **Resume/skip:** pre-insert matching `oracle_results` rows into a temp
   SQLite DB; assert the oracle call is skipped for that group and executed
   under `force=True`.
3. **Failure isolation:** stub raises `RuntimeError` for one group; assert the
   remaining groups still run and the failure summary lists exactly one entry.
4. **Report join:** fixture DB with known oracle/base rows (including a
   strategy present only in oracle, one only in base, one below signal
   threshold, one negative base sharpe); assert `viable` flags, NaN handling,
   column order, and sort order.
5. **signals_per_week:** `_period_to_weeks("6m")`, `"1m"`, `"2w"`, `"1y"`
   asserted to hand-computed values (1e-6); division verified in a report row.
6. **CSV/table:** `viability_report` table row count matches DataFrame; CSV
   written with exact header.

Runtime verification (manual, post-implementation): 
`uv run ./strategy/kairos_pipeline.py --stage auto --intervals 1d --backtest_period 1m --asset_class crypto`
must complete end-to-end on this machine (GPU present) and produce a report CSV
whose viable rows all satisfy the thresholds; re-running immediately must skip
all completed groups; `--report_only` must reproduce the same CSV from the DB.

## 6. Implementation order

1. `_period_to_weeks` helper + tests (smallest, everything depends on it).
2. `build_viability_report` + `viability_report` table + tests (works from a
   fixture DB before the auto stage exists).
3. `run_stage_auto` chaining + resumability + failure isolation + tests.
4. CLI wiring (`--stage auto`, `--intervals`, `--min_sharpe`, `--min_signals`,
   `--force`, `--skip_universe`, `--report_only`) + interval-awareness check of
   the correlation stage.
5. PIPELINE.md update + reduced-run verification.

## 7. Constraints (per APPENDIX-A / CLAUDE.md)

- numpy/pandas/sqlite3 only; reuse `run_stage_*`, `run_backtest_subprocess`,
  `_rows_from_export`, `dump_csv`, `resolve_disabled_strategies`,
  `asset_class_for`, `_period_to_bars`'s parser, `calendar_days_for_bars`.
- Never touch `PRED_SAMPLES` / `DEMO_LOOKBACK` constants; speed knobs are CLI
  flags only.
- Existing single-stage invocations must keep working byte-identically.
