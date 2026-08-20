# Kairos Multi-Interval Rollout: 1d → 1h

**Version:** 1.0
**Date:** 2026-08-20
**Scope:** Extend every pipeline phase (universe → correlation → oracle →
base → finetuning → signal generation → papertrade/MTM → Telegram digest)
from `1d`-only to also cover `1h`, as the first step toward progressively
shorter daily-advice cadences. E0 (shared plumbing hardening) is
implemented by this document; E1–E8 are scoped but not yet built.

## 1. Motivation

Every phase of Kairos has only ever run on the `1d` interval. The goal is
daily advice messages on a progressively shorter cadence. `1h` is the next
target — not literal "12h" — because neither yfinance nor
`exchange_calendars` (via `kairos/calendars.py`'s `_INTERVAL_MINUTES`, which
tops out at `1h`/60min) recognize a native 12h or 3h bar. `1h` already has a
head start: a documented playbook (`docs/playbooks/hourly-signals.md`), a
`--include-hourly` flag in `kairos_weekly_discovery.py`, and interval-keyed
predcache/finetune-registry plumbing already in place.

A codebase survey (three parallel research passes, 2026-08-20) found the
interval plumbing broader than expected: `KairosSettings.interval`,
`--interval`/`--intervals` CLI flags, `kairos_predcache.make_key()`, the
`finetuned_models` registry (`UNIQUE(assets, interval)`), `oracle_results`/
`model_results`/`disabled_strategies` tables, and `kairos_signals.run()`'s
per-`(assets, interval)` grouping are all already interval-aware. The real
gap is narrower than "rebuild everything twice": a handful of genuine bugs
where a value was silently daily-only, plus stages that need actually
running (not rewriting) against `1h` data, plus per-interval recalibration
of statistical thresholds tuned by eye for daily bars.

Universe screening and correlation grouping are to be **fully recomputed
from native 1h bars**, not reuse 1d-derived liquidity stats — liquidity and
correlation structure can genuinely differ intraday, and this is the
literal reading of "repeat every phase."

Existing `1d` config/filters/finetuned models must keep working unchanged
throughout. This is an additive parallel track keyed by `interval`, not a
migration.

## 2. Roadmap (Epic/Story breakdown)

Each epic ends with a short instructions doc under `docs/playbooks/`
(mirroring `daily-signals.md`/`hourly-signals.md`'s style): command to run,
what "success" looks like, what to check if it isn't — written so a Haiku
model or a foggy-headed dev can execute it mechanically.

### E0 — Shared plumbing hardening (DONE, this document)

Four cross-cutting bugs that would silently corrupt results the moment two
intervals coexist:

1. **Sharpe annualization hardcoded `sqrt(252)`** (a daily-bar assumption)
   in `kairos_orchestrator.py` (`_compute_shadow_performance`,
   `_build_results`), `kairos_strategies.py` (`compute_metrics`), and
   `kairos_backtest.py` (`BacktestEngine._compute_metrics`). Fixed by a new
   shared `bars_per_year(interval)` helper in `kairos_backtest.py`
   (`BARS_PER_DAY[interval] * 252`, matching the existing convention of
   annualizing against 252 trading days for both equities and crypto), used
   everywhere `sqrt(252)` used to be hardcoded. `interval="1d"` still
   resolves to exactly `252`, so 1d Sharpe values are bit-identical to
   before. `KairosOrchestrator._build_results`'s per-strategy `ssharpe`
   already derived its own empirical trades-per-year from trade dates
   (`tpy = len(strades_sorted) * 365.0 / span`) — left untouched, it was
   already interval-correct.
2. **Three duplicated `bars_per_day` dicts** (`kairos_pipeline.py`,
   `kairos_strategies.py` ×2) — identical today but a maintenance hazard.
   Consolidated into one `BARS_PER_DAY` dict in `kairos_backtest.py` (the
   common module both `kairos_strategies.py` and `kairos_orchestrator.py`
   already import from, avoiding an import cycle with `kairos_strategies.py`
   since it itself imports `kairos_orchestrator.py`). `kairos_strategies.py`
   and `kairos_pipeline.py` now import it rather than redefining it.
3. **`_signals_cache_key()` truncated to `as_of.date()`** in
   `kairos_signals.py` — correct for `1d` (one fetch/day) but for `1h` this
   collapsed every hourly signal-generation call in a day onto one cache
   key, serving stale intraday data for the rest of the day. Fixed with a
   new `_cache_as_of_value(now, interval)`: daily-or-coarser intervals
   (`kairos.calendars._DAILY_OR_COARSER`) still truncate to a bare
   `date` — byte-identical to the old behavior, preserving the
   documented cross-process cache-hit property for `1d` — while intraday
   intervals floor `now` to the current bar boundary (via
   `_interval_to_timedelta`), so a freshly-closed bar busts the cache
   instead of colliding with the rest of that calendar day.
4. **Correlation-stage `MAX(run_id)` ignored interval.** Two spots in
   `kairos_pipeline.py`: `run_stage_correlation`'s universe-survivors query,
   and `_group_symbols_from_db` (used by `--stage oracle/base/finetuned
   --group_id`). Both used to grab the single most-recently-run
   `universe_screen`/`suggested_groups` row-set regardless of what interval
   was actually requested — harmless only because `1d` was the only
   interval that ever existed. Both now join through the `runs` table
   (`stage`/`interval` columns, already populated by `start_run`) to find
   the latest run *for the requested interval specifically*. `--stage auto`'s
   own skip-detection queries were already correctly interval-scoped this
   way — no change needed there.

**Verification:** `uv run --with pytest python -m pytest tests/unit/ -q` —
1662 passed, 1 failed. The one failure
(`TestRunStageFinetuneNextNotifications::test_no_telegram_flag_suppresses_all_notifications`)
is confirmed pre-existing on `master` (reproduced identically with these
changes stashed out) — an unrelated Telegram-notify-gating bug in
`finetune_next`, not touched by this work. New regression coverage added:
`TestBarsPerYear` (`test_sharpe_safety.py`), three `_cache_as_of_value`
tests (`test_signals_report.py`), and
`test_correlation_does_not_leak_across_interval_universe_runs`
(`test_pipeline_auto.py`) plus fixture fixes to the four correlation tests
that were bypassing `start_run` and got legitimately caught by fix #4.

### E1 — Universe stage for 1h

`run_stage_universe` (`kairos_pipeline.py`) hardcodes `interval="1d"` for
its actual liquidity/volatility fetch; the `--interval` argument today only
drives a bolt-on existence probe (`interval_probe_ok`) that's computed but
never gates pass/fail. Rework: fetch and compute liquidity/dollar-volume/
ATR% from native `1h` bars when `interval="1h"`; scale `compute_universe_stats`'s
`ann_vol` annualization the same way as E0 item 1 (via `bars_per_year`);
recalibrate `evaluate_liquidity`'s thresholds for 1h bar economics (dollar
volume per 1h bar is much smaller than per 1d bar — needs its own
threshold); turn `interval_probe_ok` into a real pass/fail gate.

### E2 — Correlation stage for 1h

Already interval-parameterized once E0 item 4 landed. Verify
`min_overlap=150`/`roll_window=30` (bar counts) still make sense at 1h
cadence (150 bars ≈ 6 days vs 150 days at 1d) — likely needs its own tuned
bar-count constants for 1h rather than reusing the 1d numbers.

### E3 — Oracle stage for 1h + `OrchestratorConfig` calibration

Oracle already forwards `--interval` cleanly. Add an interval-keyed
`OrchestratorConfig` preset mechanism (mirroring `_DISABLED_BY_CLASS`'s
`(interval, asset_class)` keying) for `entropy_threshold`/`kurtosis_max`/
`min_volume_percentile` — finer bars produce noisier per-bar distributions,
so the 1d-calibrated thresholds likely don't transfer as-is. Calibrate via
a real `debug_filters=True` sweep over 1h data.

### E4 — Base model backtest for 1h

Mostly "run it and verify" — `run_backtest_subprocess` and
`refresh_disabled_strategies` already forward/key by interval correctly.

### E5 — Finetuning loop (`finetune_next`) for 1h

Registry is already interval-safe (`UNIQUE(assets, interval)`, checkpoint
dirs named `{interval}__{assets}/`). Verify `_YF_MAX_DAYS`/
`compute_finetune_periods` respect yfinance's 729-day cap on 1h history
(already flagged as a live caveat in `hourly-signals.md`).

### E6 — Signal generation + selection/allocation for 1h

`kairos_signals.py` already groups by `(assets, interval)`;
`signal_selection.py`/`allocation.py` are interval-agnostic by construction
(operate on trade-count stats, not calendar time). Mostly verification once
E0 item 3 is exercised live.

### E7 — Papertrade/MTM/margin for 1h

`--interval` already threads through `kairos_papertrade.py`'s step/
`floor_dt` generically. But the day-loop's MTM/margin logic is
unconditionally daily (`day_start`/`day_end` always a 24h window, one
`kairos_mtm_daily` row per date, `daily_financing()`'s `/360` accrual)
regardless of `--interval`. Plan: keep MTM/margin snapshotting at daily
cadence even when signals generate hourly — margin doesn't need
finer-than-daily marking, and it avoids re-deriving the financing-accrual
math. Needs a guard so `daily_financing()` can't be invoked more than once
per calendar day if the loop step becomes `1h` instead of `1d`.

### E8 — Hourly Telegram digest + scheduling

New wrapper mirroring `scripts/kairos_daily_signals.py` for `1h` (or extend
it, matching `kairos_weekly_discovery.py`'s `--include-hourly` pattern),
with notify-on-signal-only — `hourly-signals.md` already flags this as the
highest-value automation gap. New systemd timer on an hourly `OnCalendar`.

## 3. Non-goals (this document)

- No change to `1d` behavior anywhere — every E0 fix is byte-identical for
  `interval="1d"`, verified by the full test suite.
- No literal 12h/3h support — not natively fetchable from yfinance or
  resolvable by `exchange_calendars`; out of scope unless a future
  synthetic-resampling layer is explicitly requested.
- E1–E8 are scoped, not implemented, by this document.
