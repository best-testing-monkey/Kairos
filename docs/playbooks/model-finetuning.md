# Model finetuning (idle-time routine)

Automatically finetune a group-specific Kronos model for the single best
not-yet-finetuned `(asset group, interval)` candidate, backtest it, and
accept or reject it against the existing base-model result. Designed to be
run repeatedly whenever the machine is otherwise idle — one candidate per
invocation, so a menagerie of specialized models builds up incrementally
over time. Full stage reference: [`strategy/PIPELINE.md`](../../strategy/PIPELINE.md#stage-finetune_next-automated-finetuning-and-comparison).

## Prerequisites

- GPU available. Both halves of this routine need it: training runs
  `uv run finetune ... --device cuda` unconditionally, and the backtest half
  reuses the same GPU-bound `base`/`finetuned` stage machinery as the rest of
  the pipeline.
- At least one completed [weekly strategy discovery](weekly-strategy-discovery.md)
  run (`--stage auto`) for the interval you care about — candidate selection
  requires both an `oracle_results` row set *and* an existing `stage='base'`
  `model_results` row for the same `(assets, interval, backtest_period)`
  triple. No base run for a profile means it can never be selected.
- Run this **soon after the weekly refresh**, not days later. The finetuned
  backtest's test window is re-anchored to "now" at run time; the base run
  it's compared against was anchored to whenever the weekly last ran. The
  closer together the two are, the more comparable (and fair) the accept/
  reject verdict.

## When to run

Any idle GPU window after the weekly refresh: overnight, over a weekend, or
just whenever nothing else needs the card. There's no harm in running it
back-to-back — each invocation claims exactly one candidate and permanently
removes it from consideration (accepted, rejected, or failed), so repeated
runs work through the candidate list one at a time rather than repeating
themselves.

## Steps

### 1. Preview with `--dry_run`

Always sanity-check first — this prints the selected candidate, the
computed train/test periods, and the exact planned training command, with
**zero side effects** (no DB row inserted, no directory created, no
subprocess run):

```bash
uv run ./strategy/kairos_pipeline.py --stage finetune_next --dry_run
```

Expect output like:

```
[finetune_next] candidate: assets='BTC-USD,ETH-USD,SOL-USD' interval=1h backtest_period=3m viable_count=5 mean_sharpe=1.23
[finetune_next] periods: {'train_start': '...', 'train_end': '...', 'test_start': '...', 'test_end': '...'}
[finetune_next] planned training command: uv run finetune --model NeoQuasar/Kronos-base --symbol BTC-USD ETH-USD SOL-USD --interval 1h --start ... --end ... --device cuda --epochs 10 --batch-size 32 --output-model models/finetuned/1h__BTC-USD_ETH-USD_SOL-USD
[finetune_next] --dry_run: no side effects.
```

If it instead prints `no candidates found`, there's nothing left to
finetune right now — every profile with oracle + base data has already been
claimed (accepted/rejected/failed), or no profile yet has both.

### 2. Run for real

```bash
uv run ./strategy/kairos_pipeline.py --stage finetune_next
```

This does five things in order:

1. Selects the top candidate — the not-yet-finetuned `(assets, interval)`
   profile with the most oracle strategies passing `sharpe > 0 AND
   signal_count >= 3` ("viable-bar" strategies), tie-broken by their mean
   Sharpe. Requires an existing `stage='base'` backtest for the identical
   profile.
2. Inserts a `finetuned_models` registry row with `status='training'`,
   immediately claiming that `(assets, interval)` slot so a concurrent or
   later run can't pick the same candidate.
3. Trains a group-specific Kronos model: a multi-symbol `uv run finetune`
   subprocess (tokenizer frozen, base model `NeoQuasar/Kronos-base`) over
   all assets in the group at once, saving `best_model/` and `final_model/`
   checkpoints.
4. Backtests the trained checkpoint with `stage='finetuned'`, using
   parameters **identical** to the last base run for this profile (same
   assets, interval, backtest_period, pred_samples).
5. Compares finetuned vs. base and writes the verdict (see below).

### 3. Read the verdict

The run ends with a block like:

```
[finetune_next] VERDICT: ACCEPTED
  assets=BTC-USD,ETH-USD,SOL-USD interval=1h backtest_period=3m
  base: viable_count=5 mean_sharpe=1.1000 (run_id=42)
  ft:   viable_count=7 mean_sharpe=1.3500 (run_id=57)
  model_path=models/finetuned/1h__BTC-USD_ETH-USD_SOL-USD/best_model
  registry id=12
```

The accept gate: the finetuned backtest is **accepted** iff it has strictly
more viable-bar strategies (`sharpe > 0`, `>= 3` signals) than the base run,
with ties broken by mean Sharpe of those strategies. Otherwise it's
**rejected**. A non-zero exit from the training subprocess marks the
candidate **failed** instead (backtest never runs) — the partial model
directory is kept for post-mortem.

Rejected and failed candidates are **never automatically retried** — they
stay in `finetuned_models` under that status permanently, so future
`finetune_next` runs skip them and move on to the next-best candidate. See
"Re-queuing a candidate" below to force a retry.

## Where models and the registry live

Models: `models/finetuned/{interval}__{SORTED_ASSETS}/`, e.g.
`models/finetuned/1h__BTC-USD_ETH-USD_SOL-USD/` — sorted, underscore-joined
asset list, so the directory name is stable regardless of the order assets
were originally passed in. Inside: `best_model/`, `final_model/` (both
written by the trainer), and `metadata.json` (a mirror of the registry row,
written after training and again after the verdict). On rejection, an empty
`REJECTED` marker file is also written to the model dir — a fast filesystem-
level check without touching the DB.

Registry: the `finetuned_models` table in `data/pipeline_results.db`, keyed
`UNIQUE(assets, interval)` on the **sorted** assets CSV. Columns: `assets`
(sorted, canonical key), `assets_raw` (as used in `oracle_results`/
`model_results`), `interval`, `backtest_period`, `train_start`/`train_end`,
`test_start`/`test_end`, `model_path`, `status`
(`training`/`accepted`/`rejected`/`failed`), `base_run_id`/
`finetuned_run_id`, `base_viable_count`/`ft_viable_count`,
`base_mean_sharpe`/`ft_mean_sharpe`, `created_at`.

Inspect it directly:

```bash
sqlite3 data/pipeline_results.db \
  "SELECT id, assets, interval, status, base_viable_count, ft_viable_count,
          base_mean_sharpe, ft_mean_sharpe, model_path
   FROM finetuned_models ORDER BY id DESC LIMIT 20;"
```

## Re-queuing a rejected or failed candidate

Manual re-queue is the only way to retry a profile once it's `rejected` or
`failed` — pass the exact assets and interval to bypass ranking:

```bash
uv run ./strategy/kairos_pipeline.py --stage finetune_next --assets BTC-USD ETH-USD SOL-USD --interval 1h
```

This deletes the existing `finetuned_models` row for that profile (matched
on the sorted-assets key) and reruns the full train/backtest/compare cycle
from scratch, ignoring the normal candidate-ranking step.

## Runtime expectations

The predictor is Kronos-base, 102.3M parameters, tokenizer frozen during
finetuning — only the predictor head trains. Training has previously run
successfully on this machine (see the pre-existing checkpoints under the
gitignored `finetune_csv/models/` directory from earlier manual finetuning
experiments), so there's a known-working path here, but budget real time:
training (`--ft_epochs 10` default, `--ft_batch_size 32` default) plus the
subsequent full backtest at `--pred_samples 100` easily runs to tens of
minutes to a few hours depending on the asset group's history length and
interval — this is exactly why it's framed as an idle-time routine rather
than something to wait on interactively.

`--ft_epochs` and `--ft_batch_size` tune the training subprocess only (both
forwarded straight to `uv run finetune --epochs`/`--batch-size`); everything
else about the backtest half is fixed to match the base run it's compared
against.

## The yfinance data horizon

Training data comes from `price_cache`, which is ultimately backed by
yfinance's own history limits. For intraday intervals (`1h`/`60m`), Yahoo
Finance only serves **729 days** of history no matter how far back
`train_start` is computed — the trainer's fetch just returns whatever
actually exists within that horizon, so 1h groups always train on a
capped ~2-year window regardless of `train_start`. Daily-ish intervals use
a 5-year horizon instead. This mirrors the same `yf_max_days` table used by
`kairos_strategies.fetch_data_raw`.

## Automation opportunities

- Nothing schedules this today — a cron job or systemd timer that runs
  `--stage finetune_next` in a loop whenever the GPU is otherwise idle (with
  a lock file or `nvidia-smi` check to avoid clobbering a manual backtest)
  would let the model menagerie grow unattended.
- **Not yet wired**: nothing downstream consumes `accepted` models today —
  `kairos_signals.py` and the viability report still only ever read
  `stage='base'` results. Wiring `finetuned_models` (status=`accepted`) into
  `kairos_signals`/viability so an accepted group actually gets used for
  live signals is the declared next step, not yet built.
- A notification (Telegram, email, or similar) on each `ACCEPTED`/`REJECTED`/
  `FAILED` verdict, so a human doesn't have to remember to check
  `finetuned_models` after an overnight run — see the Phase 3 delivery layer
  in [`ROADMAP.md`](../../ROADMAP.md), also unbuilt.

See also: [weekly-strategy-discovery.md](weekly-strategy-discovery.md), [`strategy/PIPELINE.md`](../../strategy/PIPELINE.md).
