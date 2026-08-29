# Phase 2: wiring per-(model, class) stats into selection, allocation and margin

Phase 1 (done) records the stats. **Nothing consumes them yet.** This document is
the deliverable that says exactly what must change to make them matter, with call
sites already traced so the next session doesn't have to re-derive them.

## What phase 1 built

| Thing | Where |
|---|---|
| `strategy_class_stats` table — one row per (run, strategy, asset class), carrying `model_path` | `strategy/kairos_pipeline.py` `SCHEMA` |
| Canonical per-symbol classifier `asset_class_of_symbol()` → `equity｜crypto｜fx_commodity` | `strategy/kairos_backtest.py` |
| Per-class aggregation at source (exact: each signal attributed to its own symbol's class) | `kairos_orchestrator._compute_shadow_performance{,_naive}` → `self._shadow_performance_by_class` |
| Read helper with corpus fallback | `kairos_pipeline.strategy_class_stats()` |
| Backfill of the 111,657 historical rows | `scripts/backfill_class_stats.py` |

Read the stats with:

```python
kp.strategy_class_stats(conn, stage="base", asset_class="crypto",
                        model_path=None, min_signals=30)
# -> {strategy: {sharpe, win_rate, avg_pnl_per_trade, signal_count, n_groups, source}}
# source is "class" or "corpus" — it tells you whether the fallback fired.
```

## Two rules that must not be broken

1. **Never reconstruct a corpus figure from per-class rows.** Sharpe is a ratio and
   does not recombine across classes. `asset_class=None` reads the corpus number
   from `oracle_results`/`model_results`, which is the exact pooled value. There is
   a test pinning this (`test_read_helper_corpus_mode_reads_the_corpus_table_not_the_class_table`).
2. **`asset_class` means four different things in this codebase.** Do not join on it.
   - `asset_class_of_symbol()` — 3-way, per symbol, suffix-based. Stats only.
   - `kairos_strategies.asset_class_for()` — 5-way (`fx` and `commodity` split
     separately, plus `mixed`), per GROUP by majority vote. **Load-bearing for the
     live `_DISABLED_BY_CLASS` safety net — do not "unify" it without doing item 2 below.**
   - `kairos_pipeline.asset_class_of()` — 3-way but membership-based; returns
     `unknown` for anything not in `CANDIDATE_UNIVERSE`.
   - `kairos_margin.classify_symbol()` — 7-way margin schedule, leverage math only.

## The work

### 1. Make the disabled-strategy gate model-aware

`resolve_disabled_strategies(interval, assets)` — `strategy/kairos_strategies.py:964`,
called from `strategy/kairos_signals.py:811,820`.

It takes no model identity. `kairos_signals.run()` calls it **identically** for the
base pass and the finetuned overlay pass of the same group, so a strategy disabled
for a group is disabled under every model — even though phase 1's own data shows
which strategies work differs by model. Add `model_path` and consult
`strategy_class_stats` for that model.

### 2. Replace the hand-curated class fallback

`_DISABLED_BY_CLASS` — `strategy/kairos_strategies.py:906-952`. A dict hand-derived
from a single 2026-07-05 sweep, and today the *only* class-level gate in the live
path. It fires whenever a group has no oracle-tested DB profile.

Replace with a query against `strategy_class_stats`. **Careful:** it is keyed 5-way
(`("1d","fx")`, `("1d","commodity")` separately) while the new stats are 3-way
(`fx_commodity`). Reconcile deliberately — if `asset_class_for()` starts returning
`fx_commodity`, every one of those keys silently stops matching and the fallback
returns an empty disabled set, letting strategies as bad as `volume_fade`
(−150 mean Sharpe on crypto) run unfiltered. That is a live safety regression, and
it is why phase 1 left `asset_class_for()` alone.

### 3. Stop discarding the class at signal time

`viability_report.asset_class` is already computed and stored
(`kairos_pipeline.py:2358`) but is **not** in `STATS_COLUMNS`
(`kairos_signals.py:381-386`), so it is dropped before reaching `stats_row`
(`:920-946`) and never reaches `Candidate` (`allocation.py:224`). Thread it through.

### 4. Let selection rules filter on class

`strategy/signal_selection.py:74-99` — the DSL column registry has no
`Asset Class`. Add it to `_TEXT_COLUMNS` so rules like
`"'Asset Class' == 'crypto', 'Sharpe' > 1.0, TOP 3"` become expressible. Depends
on item 3.

### 5. Source allocation priors per (model, class)

`strategy/allocation.py` `compute_derived()` (`:121`) shrinks toward a no-edge prior
using `n` and `base_win_rate`, which arrive from `viability_report` per GROUP.
Source them per (model, class) instead so a strategy's crypto prior is not
contaminated by its equity record. `AllocationConfig.n0=100` / `min_n=50` are
global constants — consider whether they should vary by class, given per-class
signal volume differs by an order of magnitude.

### 6. Decide whether margin should care

`admission_check()` (`kairos_mtm.py:252`) takes ticker, notional and account state.
**Nothing per-strategy feeds in today.** If class-aware strategy quality should
influence sizing, this is where it goes — but it is a genuine product decision,
not a mechanical wiring job, and needs its own thinking.

### 7. Point the analysis scripts at the table

`docs/papers/analyze_by_market3.py:22,28,60` reimplements per-class stats as a
read-time group-majority heuristic with the same 3-way taxonomy — it is the prior
art that motivated this work. Its weakness is exactly what phase 1 fixes: it
attributes a whole group to one class, so a lone crypto symbol in a mostly-equity
group is either mis-attributed or dropped as mixed. Repoint it at
`strategy_class_stats` and the papers gain exact per-class numbers, including for
the ~1.3% of groups that are genuinely mixed.

## Calibration still owed

`CLASS_STATS_MIN_SIGNALS = 30` (`kairos_pipeline.py`) is a starting default, not a
calibrated one — there was no per-class history to tune it against when it was
written. For scale: `AllocationConfig.min_n` is 50, `n0` is 100,
`refresh_disabled_strategies` uses 5, the viability report 3. Tune it against real
per-class volume before anything live depends on the fallback boundary.

## Known limitation carried from phase 1

Backfilled rows (everything before this change) derive their class from **group
composition**, not per-signal attribution — the per-symbol breakdown was discarded
before those rows were persisted. Groups spanning more than one class are marked
`'mixed'` and are invisible to per-class reads: 1,503 rows, 1.3% of the backfill.
Re-sweeping those groups is the only way to split them. New sweeps are exact.
