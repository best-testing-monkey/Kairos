# Hourly finetuning (`--stage finetune_next --interval 1h`)

Automated finetune-and-compare cycle for `1h` — the fifth stage in the
`--interval 1h` pipeline, after universe, correlation, oracle, and base. Same
mechanics as [model-finetuning.md](model-finetuning.md) (registry, lock,
GPU-idle check, verdict logic, notifications) — this page covers only what
differs for `1h`, plus a real bug found while verifying it.

## ⚠️ Known bug: automated candidate selection is interval-blind (found 2026-08-20, unfixed)

`--stage finetune_next --interval 1h` with **no `--assets`** currently finds
**zero candidates** for almost every asset, even ones with fresh `1h` oracle
and base results sitting in the DB. Root cause, in
`select_finetune_candidate` (`strategy/kairos_pipeline.py`, ~line 1322):

```python
already_registered = {
    r[0] for r in conn.execute("SELECT assets FROM finetuned_models").fetchall()
}
```

This query has **no `interval` filter** — it collects every `assets` string
ever registered in `finetuned_models`, across *all* intervals, and then (line
1329/1357) excludes a candidate if its `assets_sorted` is anywhere in that
set. Since `finetuned_models` has a `UNIQUE(assets, interval)` constraint —
i.e. it's *designed* to hold independent rows per interval for the same asset
combination — this check defeats that design: once an asset combination has
**any** `1d` registry row (training/accepted/rejected/failed), it's
permanently invisible to candidate selection at `1h` (or any other interval),
forever, with no way to retry short of a manual `--assets` override.

Observed live: `ZW=F` has real `1h` oracle (`run_id=737`) and base
(`run_id=738`) results, but `--stage finetune_next --interval 1h --dry_run`
(no `--assets`) printed `[finetune_next] no candidates found` — because
`ZW=F|1d|failed` already exists in the registry from earlier `1d` work. Given
how much `1d` finetuning history already exists in this DB, this bug likely
blocks auto-selection for the *majority* of assets that would otherwise be
`1h` candidates.

**Workaround used to verify the rest of this story** (also the only way to
run `1h` finetune_next at all until this is fixed): pass `--assets` explicitly
to bypass `select_finetune_candidate` entirely — `run_stage_finetune_next`'s
docstring already documents this as the intended manual-re-queue escape
hatch (line ~1532: *"Select the top not-yet-finetuned candidate ..., or use
the explicitly supplied (assets, interval) for a manual re-queue"*), and it
works correctly — the actual training/registry-write path (unlike the
selection query) IS properly interval-scoped.

**Not fixed by this story** (verification-only, per its ticket) — needs a
follow-up: the fix is almost certainly adding `interval = ?` to the
`already_registered` query (and its `interval` needs threading into the
query), mirroring the same class of bug already fixed once this session in
`run_stage_correlation`'s `MAX(run_id)` query (see `docs/tickets/E11-S03-*`).

## Steps (with the workaround)

```bash
# Preview first — always.
uv run ./strategy/kairos_pipeline.py --stage finetune_next --interval 1h \
  --assets <SYMBOL> --backtest_period <period matching an existing base run> --dry_run

# Then for real. --skip-idle-check is only safe if you've independently
# confirmed via nvidia-smi that the GPU is actually idle (low memory.used) —
# the default idle check can false-positive on a laptop GPU's driver-noise
# utilization readings even with nothing running.
uv run ./strategy/kairos_pipeline.py --stage finetune_next --interval 1h \
  --assets <SYMBOL> --backtest_period <period> [--ft_epochs N] [--skip-idle-check]
```

**`--backtest_period` must match an existing `1h` base run's period** for the
comparison step to find a baseline — `run_stage_finetune_next`'s own default
(`6m`) will NOT match a base run done with a different period (e.g. this
page's own verification run used `1m`, matching E13-S01's base run).

## What a successful run looks like (observed 2026-08-20, `ZW=F`, `1m` period, `--ft_epochs 1`)

```
[finetune_next] candidate: assets='ZW=F' interval=1h backtest_period=1m viable_count=None mean_sharpe=None
[finetune_next] periods: {'train_start': '2024-07-22', 'train_end': '2026-07-21', 'test_start': '2026-07-21', 'test_end': '2026-08-20'}
[finetune_next] planned training command: uv run finetune --model NeoQuasar/Kronos-base --symbol ZW=F --interval 1h --start 2024-07-22 --end 2026-07-21 --device cuda --epochs 1 --batch-size 32 --output-model .../models/finetuned/1h__ZW=F
[finetune_next] registered id=177 status=training model_dir=.../models/finetuned/1h__ZW=F
...
Stage finetuned done: built 127, disabled 25, evaluating 102 strategies (15 fired at least one signal). run_id=739.
[finetune_next] VERDICT: ACCEPTED
  assets=ZW=F interval=1h backtest_period=1m
  base: viable_count=7 mean_sharpe=7.8122 (run_id=738)
  ft:   viable_count=7 mean_sharpe=18.0641 (run_id=739)
  model_path=.../models/finetuned/1h__ZW=F/best_model
  registry id=177
```

Independently verified (not just trusting the log):
- `train_start` (`2024-07-22`) to `train_end` (`2026-07-21`) is exactly the
  729-day yfinance `1h` history cap (`_YF_MAX_DAYS["1h"] = 729`) — confirmed
  working correctly.
- `models/finetuned/1h__ZW=F/` exists with real `best_model/`/`final_model/`
  subdirs, each containing `config.json`/`model.safetensors`/`README.md`.
- `SELECT * FROM finetuned_models WHERE id=177` matches the printed verdict
  exactly: `interval='1h'`, `status='accepted'`, `base_run_id=738`,
  `ft_run_id=739`, `model_path` ends in `/best_model`.

**Runtime**: with `--ft_epochs 1` (reduced from the default `10` purely to
verify the machinery quickly, not for production quality), the full cycle —
GPU-idle check, training subprocess, finetuned backtest, comparison, registry
write — took under 4 minutes end to end. Real production runs should use the
default epoch count and budget accordingly longer (untimed here, but see
[model-finetuning.md](model-finetuning.md)'s "Runtime expectations" for the
`1d` baseline to extrapolate from — `1h` trains over more bars per calendar
day at the same `729`-day/`5`-year history-length difference, so expect it to
land somewhere between `1d`'s numbers and a naive per-bar-count scaling).

## Caveats

- **The candidate-selection bug above is the main blocker** to running this
  stage unattended for `1h` — until fixed, every invocation needs an explicit
  `--assets`.
- Telegram notifications work identically to `1d` (see
  [model-finetuning.md](model-finetuning.md)'s "Notifications" section) —
  source `~/.config/kairos/kairos.env` first or they silently no-op.
- Same benign LAPACK `DLASCL` warning seen in
  [hourly-oracle.md](hourly-oracle.md)/[hourly-base-model.md](hourly-base-model.md)
  appeared again in this run's base-comparison sub-step; harmless.

See also: [hourly-base-model.md](hourly-base-model.md) (prerequisite stage,
provides the comparison baseline) and
[model-finetuning.md](model-finetuning.md) (shared mechanics, `1d`-focused).
