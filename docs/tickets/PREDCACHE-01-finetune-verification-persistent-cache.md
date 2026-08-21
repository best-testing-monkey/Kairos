# PREDCACHE-01: Cache finetune-verification predictions persistently; purge on rejection

## Context

Investigated 2026-08-21 while reviewing results from the 1h finetune
catch-up batch (see `docs/handoff-202608211629.md` and later handoffs for
the batch itself). Baz asked a precise question: do the predictions
computed while verifying a freshly-finetuned model (the backtest step
inside `run_stage_finetune_next`) land in the same persistent prediction
cache that `kairos_papertrade.py` uses? Traced the code to answer
definitively: **no.**

- `run_stage_finetune_next`'s call to `run_stage_model(stage="finetuned",
  ...)` (`strategy/kairos_pipeline.py` around line 1815) passes no
  `extra_env`.
- `run_stage_model` forwards `extra_env` straight to
  `run_backtest_subprocess`, which only builds a custom subprocess `env`
  when `extra_env` is truthy (`strategy/kairos_pipeline.py:948-950`) --
  otherwise `env=None`, meaning the subprocess inherits the parent's plain
  environment.
- `KAIROS_PRED_CACHE_DIR` is never set anywhere in the `finetune_next` code
  path, and isn't set in `~/.config/kairos/kairos.env` either.
- `kairos_predcache`'s cache is strictly **opt-in** -- it only activates
  when `KAIROS_PRED_CACHE_DIR` is set (`strategy/kairos_predcache.py:29`).
  Only `kairos_papertrade.py` turns it on, via its own
  `_ensure_pred_cache_dir_env()` at the top of `main()`.

Net effect: every finetune verification backtest recomputes every
prediction from scratch and throws them away when the subprocess exits.
The same is true for a model's **base** backtest (`stage="base"`, same code
path). This is pure wasted GPU/compute time whenever the same
symbol/date/model combination gets predicted again later (e.g. a live
papertrade run against a model that was just verified, or `--stage auto`'s
own base-stage backtest for a profile that gets finetuned soon after).

Separately confirmed while answering the same question:
`kairos_signal_replay.py --precompute` reads exclusively from the
`signals_cache` table (populated only by `kairos_signals.py`'s `run()`,
the live/daily signal-generation entrypoint) -- a completely different
table from `oracle_results`/`model_results` that `finetune_next`
populates. So even if predictions *were* cached during verification,
`signal_replay` still wouldn't see them; that's a separate, already-known
limitation (see `docs/playbooks/hourly-discovery-sweep.md`'s persistence
guidance), not something this ticket needs to fix.

## Goal

1. Thread `KAIROS_PRED_CACHE_DIR` (pointed at the same persistent
   `data/predcache/` directory `kairos_papertrade.py` uses, via
   `DEFAULT_PRED_CACHE_DIR` in `strategy/kairos_papertrade.py`) through
   `run_stage_finetune_next`'s calls to `run_stage_model` for **both**
   `stage="base"` and `stage="finetuned"` (search
   `run_stage_finetune_next` for every `run_stage_model(` call site --
   there is more than one, base and finetuned are separate calls).
   Predictions computed during verification then land in the same shared
   cache papertrade already warms and reuses, so a later live papertrade
   run (or `--stage auto` run) against the same
   symbol/interval/date/model/checkpoint-fingerprint gets a cache hit
   instead of recomputing.
2. **On rejection**, purge that model's cached predictions from the
   persistent cache -- a rejected checkpoint is never used again
   (`select_finetune_candidate` permanently excludes rejected/failed
   profiles from auto-select; the checkpoint's disk weights are already
   deleted on rejection as of the `strategy/kairos_pipeline.py` fix landed
   alongside this ticket, see `run_stage_finetune_next`'s reject branch),
   so cached predictions keyed to that checkpoint's `model_path` are dead
   weight, taking up disk/memory budget in `kairos_predcache` for nothing.
   `kairos_predcache.PredictionCache` currently has no purge/delete API at
   all (`get`/`has`/`put` only) -- add one, e.g. `purge_model(model_id: str)`
   that removes every sqlite row, in-memory LRU entry, and (if any remain)
   legacy on-disk `.npz` file whose cache key's `model_id` component
   (see `make_key`'s parameters, `strategy/kairos_predcache.py:61`) matches
   the rejected `model_path`. Call it from `run_stage_finetune_next`'s
   reject branch right where the weight directories get deleted.

## Non-goals

- Do not change anything about `--stage auto`'s own separate ephemeral
  tempdir use of `kairos_predcache` (per `CLAUDE.md`'s existing note, that
  usage is for a different reason -- reusing predictions across overlapping
  correlation groups within one run -- and is unaffected by this ticket).
- Do not change `kairos_signal_replay.py`'s data source (`signals_cache`);
  that's a separate, already-understood limitation, not a bug to fix here.

## Acceptance criteria

- A unit test proving `run_stage_finetune_next`'s calls to
  `run_stage_model` (both base and finetuned stages) pass
  `extra_env={"KAIROS_PRED_CACHE_DIR": <the real persistent dir>}` (mock
  `run_stage_model` and assert on its call kwargs, matching the pattern
  already used by `test_pipeline_auto.py`'s existing `_capture`-style
  tests).
- A unit test for `PredictionCache.purge_model()`: put a few entries under
  one `model_id` and a few under another, purge one, assert only that
  model's entries are gone (sqlite rows AND the in-memory LRU) while the
  other model's entries survive.
- A unit test proving a rejected `run_stage_finetune_next` call invokes the
  purge (mock/spy on `kairos_predcache`'s purge function, assert it's
  called with the rejected checkpoint's `model_path` once the reject branch
  runs) -- and that an **accepted** run never calls it.
- Live-verify: run a real `finetune_next` cycle end to end (small, cheap
  candidate) and confirm `data/predcache/` actually gains new entries
  during the verification backtest (not just base/live papertrade runs),
  then confirm a subsequent identical `--stage base` or papertrade run
  against the same profile shows cache hits, not misses, in the existing
  `_log_group_timing()`/watchdog forensics logging.
- Full `tests/unit/` suite green, no regressions to the existing
  `kairos_predcache`/`test_predcache.py` coverage.

## Notes for whoever picks this up

- `strategy/kairos_strategies._model_checkpoint_fingerprint()` already
  exists and is the established pattern for keying a local finetuned
  checkpoint's cache entries so a retrained-in-place checkpoint at the same
  `model_path` doesn't collide with stale cached predictions under the old
  weights -- reuse it, don't reinvent it.
- Disk usage from *accepted* models' cached predictions is bounded by
  `PredictionCache.max_disk_bytes` (existing eviction, unrelated to this
  ticket) -- this ticket's purge is specifically for *rejected* models,
  which the existing eviction has no special awareness of (it would
  eventually get evicted by LRU pressure anyway, but "eventually, maybe" is
  worse than "immediately, on rejection" given rejected checkpoints are
  known-dead the moment the verdict lands).
