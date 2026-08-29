# Kairos — Claude Code Notes

## Project layout

```
model/           Kronos model (transformer, tokenizer, predictor)
strategy/        Trading strategies and backtesting engine (NOT a Python package)
kairos/          Python package: adapter, calendar, data, config
tests/           pytest suite (unit + integration)
```

`strategy/` has no `__init__.py`. Tests and scripts add it to `sys.path` explicitly.
The project is managed with **uv** (`uv run ...`), not pip/python directly.

## Running things

```bash
# Run the full backtest demo (needs GPU or ~5-10s/iteration on CPU)
uv run ./strategy/kairos_strategies.py

# Run all tests (no GPU or model download needed)
uv run --with pytest python -m pytest tests/unit/ -q

# Run a specific test file
uv run --with pytest python -m pytest tests/unit/test_kairos_distribution.py -v

# Run the asset-discovery pipeline (screening/correlation/oracle/base/finetuned stages)
uv run ./strategy/kairos_pipeline.py --stage universe   # see strategy/PIPELINE.md for all stages/flags

# Offline signal replay: fast selection/allocation iteration with no GPU/live papertrade
uv run ./strategy/kairos_signal_replay.py --precompute --start 2026-08-01 --end 2026-08-07 --interval-ladder 1h,4h,1d
uv run ./strategy/kairos_signal_replay.py --replay --interval 1d --start 2026-08-01 --end 2026-08-07 --capital 200 --max-pos-pct 15 --top-k 3
```

## Known gotchas (hard-won)

### strategy/ imports
All strategy modules (`kairos_backtest`, `kairos_orchestrator`, `kairos_meta`,
`kairos_execution`, `kairos_path`, `kairos_horizon`) live in `strategy/` and import
each other by bare name. Any script or test that uses them must add `strategy/` to
`sys.path` first.

### Strategy return types
All `generate_signal()` implementations must return a `Signal` dataclass (from
`kairos_backtest`) or `None`. Returning a plain `dict` silently breaks the
`LiquidityFilterStrategy` wrapper (which accesses `.metadata`).

### scipy missing in kairos_execution.py
`LiquidityFilterStrategy.generate_signal()` calls `scipy.stats.percentileofscore`.
If `from scipy import stats` is ever removed, every strategy wrapped by the liquidity
filter will fail with `NameError` at runtime — silently returning `None` for every
signal. Always verify the import is present.

### Entropy threshold vs. Shannon entropy
`KairosDistribution.entropy()` computes **Shannon entropy** (PMF-based, range 0–ln(20)≈3.0).
The `entropy_threshold` in `OrchestratorConfig` (default 3.0) is calibrated to this.
Do NOT revert to `density=True` in `np.histogram` — that gives differential entropy
in 1/price units (~12–14 for BTC), which would block every asset.

### Kurtosis filter threshold
`kurtosis_max` defaults to 10.0 (excess kurtosis, Fisher definition, normal=0).
Do NOT lower this to 3.0 — discrete token sampling from the Kronos model routinely
produces excess kurtosis well above 3, which would silence all directional strategies.

### Percentile key format
`_compute_stats()` stores keys as `"pct_10"` (int, no decimal). Strategy
`__init__` params like `stop_pct: float = 10.0` must be cast to `int` before
formatting into keys: `f"pct_{int(self.stop_pct)}"`.

### torch.compile not supported
Python 3.13 — `torch.compile` raises `RuntimeError: Dynamo is not supported`.
Use TF32 flags instead (`torch.backends.cuda.matmul.allow_tf32 = True`).

### GPU inference
`auto_regressive_inference` in `model/kronos.py` uses `torch.autocast('cuda', float16)`
automatically. TF32 is enabled in `_ensure_model_loaded()` when CUDA is available.
CPU mode uses INT8 dynamic quantization via `torch.quantization.quantize_dynamic`.

### GPU recovery (opt-out strict CUDA mode)
`_ensure_model_loaded()` calls `kairos_gpu.ensure_cuda()` before deciding between
the GPU and CPU/INT8 branches. By default CUDA is *required*: if torch can't see
CUDA, `ensure_cuda()` shells out to `uv run scripts/gpu_recover.py` (an escalation
ladder L0 diagnose -> L1 free GPU processes -> L2 UVM reload -> L3 full module
reload -> L4 reboot+resume). Set `KAIROS_ALLOW_CPU=1` to restore the old silent
CPU fallback instead. Set `KAIROS_GPU_ALLOW_REBOOT=1` to permit the L4 reboot
step for unattended/overnight runs. If recovery heals the GPU but the *current*
process still can't see it (torch caches CUDA init state), the process exits
`75` (EX_TEMPFAIL); `kairos_pipeline.run_backtest_subprocess` retries such a
subprocess exactly once. Run `uv run scripts/gpu_recover.py --check-only` to
probe without side effects, or `--dry-run` to preview the full ladder.

### Telegram notifications
Four things send Telegram alerts: `strategy/kairos_pipeline.py --stage
finetune_next` (via the module-level `_notify` helper — 🟢 start, ❌ training
failure, ✅/⚠️ accept/reject verdict, 💥 any other unhandled crash after the
row is registered); the two standalone wrapper scripts
`scripts/kairos_daily_signals.py`/`scripts/kairos_weekly_discovery.py` (their
own actionable-signal/failure/summary alerts); `strategy/kairos_papertrade.py`
(own `_notify` helper, gated by `--no-telegram`/`args.notify` — 🟢 start,
✅ finish, 💥 unhandled crash, ⏱️ any single `kairos_signals.run()` call or
Phantom day-backtest that exceeds `_SLOW_ITERATION_THRESHOLD_SECONDS` (60s),
and 🧠 a heads-up right before `prewarm_prediction_cache()` actually loads a
Kronos model for one sweep unit — base, or one finetuned group — naming the
model and the date range it's about to cover; suppressed entirely when that
unit's whole period is already a `kairos_predcache` hit, so a fully-warm
prewarm is silent); and `strategy/kairos_gpu.py`'s `ensure_cuda()` (own
`_notify`, always attempted, no enable flag — 🔧 recovery starting, ❌
recovery failed, ✅ recovered/caller retrying). No other `--stage` of
`kairos_pipeline.py` sends anything — that's by design, not a bug, if you go
looking for a notification from `universe`/`correlation`/`oracle`/`base`/
`finetuned`. All of these send with `parse_mode=None` (plain text): dynamic
content (asset symbols, stderr/traceback tails) can contain an unbalanced
Markdown special character — including the literal underscore in
"finetune_next" itself, which alone broke a plain "starting" message in
production — and Telegram's legacy Markdown parser rejects the *whole*
message over a single one. See `docs/playbooks/model-finetuning.md`'s
"Notifications" section for the full per-message-type contract (that
playbook covers the `finetune_next`/daily/weekly trio specifically — the
papertrade and GPU-recovery notifications above aren't part of it).

`kairos.ops.send_telegram()` reads `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID` from
`os.environ` — it never reads `~/.config/kairos/kairos.env` itself. Only the
systemd units (`systemd/*.service`, via `EnvironmentFile=-%h/.config/kairos/kairos.env`)
load that file automatically. Running any of `strategy/kairos_pipeline.py
--stage finetune_next`, `scripts/kairos_daily_signals.py`, or
`scripts/kairos_weekly_discovery.py` directly from a shell will silently
fail every Telegram notification
(`OpsError` is caught and only logged as a `WARNING:` line, easy to miss in a
long training run) unless you source the file into that shell first:
```bash
set -a && source ~/.config/kairos/kairos.env && set +a
```
Also double-check the token's shape if notifications ever 404 instead of
"must be set": a real bot token looks like `<digits>:<~35 alnum chars>`, not
a copy-pasted placeholder.

### PRED_SAMPLES and DEMO_LOOKBACK
`PRED_SAMPLES = 100` and `DEMO_LOOKBACK = 300` in the `__main__` block of
`strategy/kairos_strategies.py` are hard constraints. Do not reduce them as a
performance shortcut — change the code instead.

### Backtesting performance (GPU)
The key optimization in `auto_regressive_inference`: run `tokenizer.encode` and
`model.decode_s1` once at the original batch size (3 assets), then expand to
`batch_orig × sample_count` only for the stochastic sampling step. This gives
~0.3s/iteration on GPU vs. 89s before.

### Model-major prediction prewarm (papertrade)
`kairos_strategies.py` has a single global model slot
(`bt_model`/`bt_predictor`) — loading a different `model_path` always
unloads+reloads (`_materialize_model`). `kairos_signals.run()`'s per-date
loop visits base → finetuned-group-1 → ... → finetuned-group-G → (next date)
→ base again, so naively that's `G+1` reloads *per backtest date*.
`kairos_papertrade.py`'s `main()` avoids this by running
`prewarm_prediction_cache()` before the whole `generate_and_dedupe_reports()`
loop: it sweeps **model-major** — every date for the base model, then every
date for each finetuned group's model, in `strategy/kairos_predcache.py`'s
shared cache — so each model loads exactly once for the whole run. The
date-major `run()` loop that follows then finds every `(symbol, bar, model)`
prediction already cached and never reloads at all
(`kairos_strategies.predict_all_batch` defers `_materialize_model` until
*after* the shared-cache lookup, and skips it entirely on a full hit).

**Prewarm is now a single inline-checked pass, not two (changed 2026-08-11,
cleaned up 2026-08-19).** Earlier, each unit (base, or a finetuned group) ran
a separate *check* pass first (stopping at the first cache miss) and only ran
a *load* pass if the check found one — see the now-superseded "Prewarm speed"
bullet in the dated section below, which described that design. The
2026-08-11 "caching fixes WIP" commit collapsed this into one pass per unit
(`_sweep_unit()` in `prewarm_prediction_cache()`): it calls
`kairos_strategies.is_batch_cached()` inline per `(group, date)` entry,
`continue`-ing past it when already cached instead of calling
`predict_all_batch()`. That WIP commit also left a real regression that sat
undetected until `TestPrewarmPredictionCache` was brought back in sync with
it on 2026-08-19: `_notify()` fired unconditionally at the top of every
sweep unit, regardless of whether anything was actually a cache miss —
violating this file's own "Telegram notifications" contract above
("suppressed entirely when that unit's whole period is already a
`kairos_predcache` hit"). Fixed by tracking a `notified` flag inside
`_sweep_unit()` and firing `_notify()` lazily, only right before the first
genuine miss's `predict_all_batch()` call; a unit that turns out fully
cached now prints a "skipped" line instead, same as before the WIP commit.
The periodic `gc.collect()` (`_PREWARM_GC_INTERVAL`, every 500 iterations)
lives in the single loop's `finally` clause, so it still fires on both hits
and misses — the "GC starvation... now effectively dormant" note in the
dated section below is itself stale; `gc.collect()` is not dormant.

The cache is a **persistent disk cache** at `data/predcache/`
(`DEFAULT_PRED_CACHE_DIR` in `kairos_papertrade.py`) — deliberately NOT
under `/tmp` or tmpfs, so it survives across separate `kairos_papertrade.py`
invocations and reboots. `main()`'s `_ensure_pred_cache_dir_env()` points
`KAIROS_PRED_CACHE_DIR` at it unless the caller already set that env var
(an explicit choice that's left untouched). This used to be an ephemeral
`tempfile.mkdtemp`, torn down in `finally` every run, specifically because
the cache key had no way to detect a checkpoint retrained in place at the
same `model_path`; that gap is now closed by
`kairos_strategies._model_checkpoint_fingerprint()` (size + mtime of
`model.safetensors` for local finetuned checkpoint directories; returns `""`
for HF repo ids like the base model, which aren't retrained in place),
folded into the shared cache key via `kairos_predcache.make_key()`'s
`checkpoint_fingerprint` parameter. A checkpoint retrained in place therefore
produces a different key and is treated as a fresh miss rather than served
stale, so persistence across invocations is now safe. Disk usage is bounded
by `PredictionCache.max_disk_bytes` (default 2GiB, oldest-`st_mtime`
eviction on every write that pushes the cache over budget), configurable via
the `KAIROS_PRED_CACHE_MAX_BYTES` env var. Use `--no-pred-cache` to disable
prewarm/reuse for a run if you need to debug around it (unrelated to this
persistence change — same escape hatch as before).
`kairos_predcache.make_key()`'s key also includes `pred_len` — always `1`
today (papertrade only predicts one bar ahead), but if that horizon becomes
configurable, thread the real value through `predict_all_batch` rather than
hardcoding it, or longer-horizon runs will silently collide with cached
1-bar predictions.

`kairos_pipeline.py --stage auto`'s own use of `kairos_predcache` is a
*different* ephemeral tempdir, for a *different* reason (reusing predictions
across overlapping correlation groups within one subprocess-spawning run,
torn down at the end of that same run) — it has no retrain-in-place
exposure and is unaffected by any of the above.

### Prewarm leak sources and crash fixes (2026-07-29/30)
A `--months-back 6` overnight run repeatedly froze the machine or crashed
before completing; root-causing it took several rounds. In addition to the
`base_entries`/`group_entries` DataFrame-retention leak (fixed earlier, see
commit `e2cf607`), three more accumulation sources and two crash classes had
to be fixed before a full 6-month run finished cleanly (RSS stable ~5.28GB).
**Two of the bullets below were later superseded by the 2026-08-08/11 work
further down — read that section too before assuming either fix still
applies as described:**

- **`is_batch_cached()` was mutating the shared LRU.** It called
  `PredictionCache.get()` purely for a boolean check, but `get()` always
  promotes a disk hit into `_mem` as a side effect — so every prewarm
  check-pass lookup was silently growing the in-memory cache the same as a
  real prediction would. Fixed with `PredictionCache.has()`, an
  existence-only check that never touches `_mem`. If you ever need a
  read-only cache-hit check elsewhere, use `.has()`, not `.get()`. Still
  true as of 2026-08-11, though `has()`'s own behavior picked up a new
  wrinkle that same date — see its docstring.
- ~~**`_prediction_cache` (the per-process dict in `kairos_strategies.py`,
  distinct from `kairos_predcache`'s disk-backed `PredictionCache`) is only
  cleared on a model switch** (`_prepare_model_switch`). Fixed with a hard
  5000-entry cap (`_PREDICTION_CACHE_MAX_ENTRIES`), clearing the whole dict
  on overflow.~~ **Superseded 2026-08-11: `_prediction_cache` was removed
  entirely**, not just capped — see the dated section below. Every lookup
  now goes straight through the shared `kairos_predcache`; `_shared_keys` (a
  much smaller symbol→cache-key map) replaced it.
- **GC starvation during long same-model sweeps.** `gc.collect()` previously
  only fired inside `_materialize_model()` on a model switch. CPython's
  generational GC thresholds are allocation-*count* based, not memory-size
  based, so large-but-few pandas DataFrames can sit as uncollected garbage
  for a long time with nothing triggering a collection. Fixed with a
  periodic `gc.collect()` every `_PREWARM_GC_INTERVAL` (500) iterations in
  all four prewarm check/load loops (base and finetuned). **Currently
  dormant as of 2026-08-11**: those loops' *check*-pass halves (the ones
  containing the `gc.collect()` calls) are commented out in the current
  `prewarm_prediction_cache()` — see "Prewarm is now a single inline-checked
  pass" above. Nothing calls `gc.collect()` periodically in the load loop
  that replaced them. Worth restoring if long-run RSS growth resurfaces.
- **`sqlite3.OperationalError: unable to open database file`** and
  **`OSError: [Errno 24] Too many open files`** both crashed live runs.
  Mitigated (not root-caused with full certainty — never deterministically
  reproduced outside the long run) with `kairos_signals._connect_with_retry()`
  (3 attempts, 1s/2s backoff, used by both `kairos_signals.run()` and
  `prewarm_prediction_cache()`) and `kairos_papertrade._raise_fd_limit()`
  (raises the process's soft `RLIMIT_NOFILE` to its hard cap, called as the
  first line of `main()`, best-effort/non-fatal on failure).
- ~~**Prewarm speed**: the check pass now stops at the first cache miss per
  unit... When a load pass does run, it always covers the *full* cross
  product for that unit.~~ **Superseded 2026-08-11**: there is no separate
  check pass anymore (see above) — every entry is checked and, on a miss,
  predicted inline in one pass.

### Overlapping prediction caches, three fix attempts, and a rewrite (2026-08-08/11)
Three more live `kairos_papertrade` debug runs (all `--months-back 6`,
default selection) climbed toward an 8GB memory ceiling and had to be killed
by hand before finishing even the base model's prewarm pass, each time after
a fix that turned out to only partially help. In order:

1. **`kairos_predcache._dfs_nbytes()` undercounted real memory** (commit
   `8dd5b1e`) — it summed `df.to_numpy().nbytes` (the raw float buffer
   only), not pandas' Index/DataFrame/block-manager overhead, so the
   in-memory LRU's `mem_budget_bytes` eviction compared against a number far
   below actual RSS and fired too late. Fixed with `df.memory_usage(deep=True)`.
   Real bug, verified by a regression test, but alone didn't stop growth —
   `mem_budget_bytes` itself (25% of available RAM *at construction time*)
   was still multiple GB on this box.
2. **`KAIROS_PRED_CACHE_MEM_BYTES` env var added** (commit `e208edc`) to pin
   an explicit, low ceiling instead of trusting the available-RAM-fraction
   default (mirrors `KAIROS_PRED_CACHE_MAX_BYTES` for the disk side). Tried
   at 512MB. Also didn't stop growth on its own — RSS still climbed to
   similar levels as before, which was the first strong signal that the
   in-memory LRU wasn't the dominant contributor after all.
3. **`kairos_strategies._dist_cache` had no size cap at all** (commit
   `c70941f`) — unlike its sibling `_prediction_cache` (5000-entry capped
   since 2026-07-30, see above), `_dist_cache` was only cleared on a model
   switch, which never happens during the same-model base sweep. Each entry
   is a `KairosDistribution` holding the full raw prediction sample list
   *and* a concatenated DataFrame copy *and* a stats dict, making it the
   fattest of the three then-overlapping caches. Fixed with the same
   entry-count-cap pattern as `_prediction_cache` (`_DIST_CACHE_MAX_ENTRIES`,
   5000, clear-on-overflow). Real bug, but a live retry afterward showed
   total RSS growth at matched progress was statistically the *same or
   slightly worse* than before this fix — meaning none of the three
   Python-dict-level fixes above were actually the dominant leak source.

**The actual fix (commit `4c3659f`, marked WIP — "still need to remove old
predcache"): a ground-up rearchitecture, not another cap.**
`kairos_predcache.PredictionCache` gained a `SqliteDict`-backed table
(`caches.db`, tablename `prediction_cache`) as its primary store —
`get()`/`has()`/`put()` all check/write it first now, ahead of the
in-memory LRU and the legacy `.npz`-per-key disk files (see the module's
docstring for the current three-layer order). `put()` no longer writes
through to the in-memory LRU at all (that call is commented out); the LRU
is now populated only as a `get()`-side-effect on a disk-only hit, so its
footprint is much smaller than when it was the primary write path.
Separately, `kairos_strategies._prediction_cache` was **removed entirely**
(not just capped) — every symbol now goes straight through the shared
`kairos_predcache`, with a small `_shared_keys: dict[str, str]` (symbol →
cache key) replacing it purely to avoid recomputing `_shared_cache_key()`
later in the same call. A new **in-process safety net**,
`strategy/memory_monitor_heap.py` (imported unconditionally near the top of
`kairos_papertrade.py`), runs a daemon thread that polls this process's own
RSS every 0.5s and, past 6000MB, suspends the main thread, dumps the top 15
`tracemalloc` allocation sites under this repo's own path, and hard-exits
(`os._exit(1)`) — a debugging tool that's also a production safety net,
since it stops growth well before an external cgroup/OOM kill (or an
un-contained freeze) would.

**WIP issues below are RESOLVED as of commit `ee745c7` (2026-08-13) — kept as
history, not a live task list. Verified again 2026-08-19: all 62 tests across
the three files pass, 0 failures.** The four bullets that used to live here
were all fixed by `ee745c7` ("Fix papertrade prewarm RAM blowup and hardened
OOM watchdog") without a matching CLAUDE.md update, which is exactly why this
note exists now — don't trust a "known issues, not yet fixed" bullet list in
this file without re-checking it against current `git log`/test runs first.
- `_no_data_fallback_warned` is back at module scope (`kairos_strategies.py`,
  outside `_dist_cache_put()`).
- `PredictionCache.__init__` now calls `os.makedirs(...)` before constructing
  `SqliteDict`.
- `_evict_disk_if_over_budget()`/`_disk_write()` were removed outright, not
  patched — `put()` no longer writes `.npz` files at all (sqlite-only), so
  there was nothing left for disk eviction to do. `test_predcache.py`
  documents the deletion of `TestDiskEviction` inline rather than replacing
  it with anything.
- `test_predcache.py`, `test_kairos_strategies_model_switch.py`, and
  `test_predict_all_batch_cache.py` no longer reference the removed
  `_prediction_cache`/`_prediction_cache_put` anywhere; all pass.

### Per-strategy signals cache (`signals_cache` table)
`kairos_signals.py`'s `run()` caches each strategy's rows (the output of
`_run_group`'s per-symbol predict → meta-filter → `generate_signal` →
row-build pipeline) in a `signals_cache` table in `db_path`
(`pipeline_results.db`), keyed by **strategy** (not just group) plus model,
group, as-of date, and lookback — `_signals_cache_key()`'s full key is
`(strategy_name, assets_str, interval, as_of_date, lookback, pred_samples,
min_ev_pct, model_path, checkpoint_fingerprint)`. `pred_samples`/
`min_ev_pct` are included because they change the cached value's meaning
(sampled distribution and the EV gate, respectively) even though nothing
originally asked for them explicitly; `checkpoint_fingerprint` (via
`kairos_strategies._model_checkpoint_fingerprint()`) busts the cache when a
finetuned checkpoint is retrained in place at the same `model_path`,
mirroring `kairos_predcache`'s own key. `as_of` is `now.date()`, not the raw
`now` timestamp — `fetch_data_raw` only ever consumes `as_of.date()`, so two
different `now` values on the same calendar day fetch identical data and
must key identically here too (this is also what makes cache hits actually
happen across separate process runs, since `papertrade`'s `base_now =
datetime.now()` never repeats exactly, but dates do for overlapping
backtest windows).

**Disabled strategies are never served stale.** `_run_group` builds
`strategies_by_name` (from `KairosOrchestrator`, already filtered by
`resolve_disabled_strategies`) *before* ever touching the cache or calling
`predict_fn` — a strategy that's since been disabled simply isn't in that
dict, so it always falls through to the existing "unknown strategy (not in
registry)" skip path, exactly as it would with no cache at all. The cache
key deliberately does NOT fingerprint the `disabled_strategies` table
itself; correctness comes from checking live status first, every time, not
from invalidating on a registry change. One consequence: if every strategy
in a group is either disabled or already cached, `predict_fn` (the GPU
model call) is never invoked for that group at all.

`use_signal_cache=True` is `run()`'s default; `--no-signal-cache` disables
it on the CLI (mirrors `kairos_papertrade.py`'s `--no-pred-cache` for the
sibling model-prediction cache). Rows are written with `INSERT OR REPLACE`
on `cache_key`, so a re-run of an already-cached key overwrites rather than
duplicating — table growth is bounded by unique-key space, not call count,
matching the rest of `pipeline_results.db`'s tables (no separate eviction).

### Watchdog forensics + persistent report de-dup (papertrade)
Two slow-run observability/de-dup mechanisms in `kairos_papertrade.py`:

- **Watchdog forensics.** Any single `kairos_signals.run()` call or Phantom
  day-backtest exceeding `_SLOW_ITERATION_THRESHOLD_SECONDS` (60s) sends a
  Telegram heads-up AND appends a forensic snapshot to
  `data/papertrade_watchdog.log` via `_log_watchdog_snapshot()` (own PID +
  VmRSS from /proc, plus `free -h` and `nvidia-smi` output — captured so
  the next machine freeze has evidence: multiple PIDs in the log means
  overlapping runs, one PID with climbing RSS means an in-process leak).
  Its companion `_log_group_timing()` logs per-(group, pass) lines to the
  same file with no subprocess calls — but only for groups that were slow
  (> `_SLOW_GROUP_THRESHOLD_SECONDS`) or a shared-cache MISS (unexpected
  once prewarm has covered the model/date), fired from
  `generate_and_dedupe_reports`' `on_group_timing` callback. When a long
  run drags, that log is the first place to look — it exists because the
  6-month leak hunt needed per-iteration RSS evidence, not anecdotes.
- **Persistent report de-dup.** `generate_and_dedupe_reports()` keeps its
  `seen` map in a `SqliteDict` (`report_seen.db`, table `seen_v2_<sha256>` for
  new runs). `_make_report_hash()` returns a v2 hash that covers `base_now`,
  interval, work-item groups, **and accepted-finetuned model paths**, plus a
  legacy hash for backward compatibility. A newly accepted finetuned model
  for an existing group now busts already-seen dates. Existing
  `seen_<legacy_hash>` tables are read as a fallback so an in-flight run
  doesn't suddenly regenerate everything; once a window starts writing to a
  v2 table it stays on v2. Gotchas: the filename is **CWD-relative** — run
  from the repo root or you silently get a fresh empty DB and regenerate
  everything. `base_now` is floored to the interval (`floor_dt`) before
  hashing/iterating so sub-day jitter doesn't fragment the key space.

### Configurable signal selection (`--signal-selection`)
Both `kairos_signals.py` and `kairos_papertrade.py` accept `--signal-selection
"<rule>"` (see `strategy/signal_selection.py` for the grammar/column registry,
e.g. `"'n' > 60, 'Win raw' > 0.6, ORDER 'EV raw %' DESC, TOP 3"`) to override
`strategy/allocation.py`'s hardcoded selection logic (RFC
`docs/rfc_allocation_sheet.md` §4.4). **The rule fully REPLACES the default
`min_n`/`ev_net>0` gate — it is not AND'd with it.** A rule that never checks
EV can admit a negative-EV signal; this is intentional (the rule is meant to
be the whole gate), but it means a rule missing an EV/quality condition is a
foot-gun, not a safety net. `ORDER`/`TOP` in the rule likewise override the
default `score` sort and `top_k`/`--top-n`; `--top-n` remains the fallback
top-K only when the rule has no `TOP` clause. `AllocationConfig.n0`/`min_n`
still drive the `shrink`/`ev_net` *math* even when a rule is active (only the
*gating* on them is bypassed) — the `Config:` line in the generated report
echoes the active rule string (`selection="..."`) so this isn't silently
invisible. Columns are limited to what's known pre-sizing (Ticker, Cluster,
Strategy, Dir, Entry, Stop, Target, Risk %, Reward %, b, n, Win raw, Win
shrunk, EV raw %, EV net %, Kelly raw, Score, Sharpe) — `Alloc %`/`Alloc
EUR`/`Flags`/`Advised liq %` aren't available to filter/sort on since they're
only computed after top-K selection.

### MTM margin/leverage system
`kairos_papertrade.py` can optionally simulate margin/leverage instead of
cash-only trading, gated by three CLI flags: `--margin-config` (path to a YAML
config, default `config/margin_ibkr.yaml`), `--max-leverage` (default `1.0` —
cash-only; only setting this above `1.0` turns margin math on), and
`--margin-utilization` (fraction of equity usable as initial margin, default
`0.8`). The math lives in two pure, GPU-free modules:

- `strategy/kairos_margin.py` — `load_margin_config()`/`MarginConfig` parses
  the YAML (per-asset-class initial/maintenance margin rates, overrides per
  symbol); `classify_symbol()` maps a ticker to a `MarginClass`.
- `strategy/kairos_mtm.py` — `compute_daily_snapshot()` builds a
  `DailySnapshot` (equity, margin used, utilization) from open position rows
  + a close price, entirely independent of phantom's own cash/equity
  numbers (`phantom` is source of truth for order fill/SL/TP mechanics only,
  never for margin math — see `APPENDIX-A-standards.md`). `admission_check()`
  gates new orders against margin utilization before they're placed.
  `liquidation_check()`/`daily_financing()`/`compute_daily_financing_total()`
  handle forced-exit and daily borrow-cost accrual.

`kairos_papertrade.py` persists one `DailySnapshot` per day and drives its
MTM metrics block + the HTML report's MTM panel (equity/drawdown/margin
utilization/liquidation markers) from that history.

**Three subtle bugs found via live `/verify` runs, all fixed — read before
touching same-day fill/close or admission-check code:**

- **Same-day fill/close blind spot (fixed, `24ff318`).** The day-loop's
  `corrected_cash` bookkeeping diffed positions at day boundaries; a position
  that both filled *and* closed within a single `runner.backtest()` call
  never appeared in either diff and was silently dropped from cash tracking.
- **Admission check defeated by same-day round trips (fixed, `79dbbb0`).**
  Fixing the bug above wasn't sufficient on its own — the admission-check
  margin gate also needed same-day round trips' margin contribution folded
  into the persisted snapshot, or a same-day open+close round trip could
  still be invisible to `admission_check()` for the *next* order that day.
- **Stale-signal-cache brackets (fixed, `7cc66d4`).** `close_reason='sl'`
  positions were showing up with *positive* realized P&L. Root cause: two
  price mirrors can drift apart intraday — `kairos_signals`' cached mirror
  can stop advancing while the papertrade day loop's own
  `_IntradayFallbackProvider` mirror keeps moving, so a stop/target computed
  from the stale mirror no longer bracketed the fresher fill-day price by the
  time the order actually placed. Fixed with a guard that rejects an order
  outright if its stop/target don't bracket the current price at fill time,
  rather than letting phantom silently fill it into a nonsensical bracket.

### Offline Signal Replay (`kairos_signal_replay.py`)
Fast, GPU-free tool for iterating on selection/allocation rules without
re-running a live `kairos_papertrade.py` pass. It does **not** call
`BacktestEngine.run()` (which needs a live model predictor to *generate*
signals) — signals are already decided by the time this tool runs, so it
reuses only `BacktestEngine`'s private, predictor-free `_check_exit`/
`_calculate_pnl` methods to resolve an already-known signal's outcome against
historical bars. **Unleveraged only** — asserts `max_leverage <= 1.0`; no
margin, CFD, or liquidation simulation (see
`docs/tickets/DESIGN_DOC_offline_signal_replay.md` §1/§4 for the explicit
scope boundary, and the module's own docstring/`--help` for the non-goals).
Its cost model (flat fee/slippage) diverges from phantom's live per-instrument
model, so results are directional signals for testing selection rules, not
P&L predictions — validate anything promising with a real
`kairos_papertrade.py` run before trusting it.

Two modes, both operating on `pipeline_results.db`:
- `--precompute --start <date> --end <date> --interval-ladder 1h,4h,1d`:
  unpacks `signals_cache` rows into a new `papertrade_signals` table, then
  walks a data-driven interval ladder per signal — smallest interval first,
  first one with enough bars wins (`resolve_interval_for_signal()`), not a
  calendar-day assumption, so a 1h replay and a 1d replay both work off the
  same mechanism. A signal that can't resolve on *any* rung of the ladder is
  disqualified (closure stats not computed, excluded from replay) rather than
  blocking the whole precompute run. Closure stats land in
  `papertrade_signals_closure`, keyed to also invalidate on `--engine-version`
  bumps.
- `--replay --interval <interval> --start <date> --end <date> --capital
  <float> [--max-pos-pct ...] [--top-k ...] [--signal-selection ...]`: replays
  `strategy/allocation.py`'s `allocate()` against the precomputed closures for
  a data-driven step grid (`SELECT DISTINCT as_of`, so it's interval-agnostic
  rather than daily-only) — this is the fast iteration loop; it never touches
  the GPU or phantom.

`papertrade_signals`/`papertrade_signals_closure` double as a cache:
`--precompute` is safe to re-run over the same window (`INSERT OR REPLACE`
semantics) and only recomputes what `--engine-version` or the window actually
changed.

Like every other price_cache caller in this codebase, this module must call
`price_cache.configure(remote=False, local_mirror_path=db_path)` before its
first real lookup (`_ensure_configured_db()`) — `price_cache.configure()`
defaults to `remote=True`, which needs an unreachable local PostgreSQL proxy
in this environment. This was missed initially (every unit test mocks
`price_cache.get_price_data` directly and never exercises real configuration
state) and silently produced 100% signal disqualification against real data
until fixed (`d4124ed`).

### price_cache: `no_data_tickers` no longer gates reads (fixed 2026-08, upstream)
`price_cache` (sibling repo, vendored into Kairos via `phantom_ledger`'s
submodule — see `pyproject.toml`'s `[tool.uv.sources]`) used to let a single
stale `no_data_tickers` row permanently block `get_price_data()`/
`fetch_bars_bulk_from_local()` for an entire ticker, any date range, forever.
This silently returned `None`/omitted tickers with hundreds of thousands of
genuinely cached rows. Fixed at the source (price_cache commit `b6f990d`,
propagated through `phantom_ledger`'s submodule bump and this repo's
`uv.lock`) — `no_data_tickers` is now a diagnostic-only, TTL'd audit trail
that neither function consults. If you see old references (docs, comments,
memory) describing "delisted ticker" as a reason `price_cache` returns no
data for a range that should be cached, that description is stale; see
`price_cache/README.md`'s `no_data_tickers` schema section for the current
behavior. `kairos_signal_replay.py`'s real-data disqualification rate was the
symptom that surfaced this bug in this repo.

### price_cache: DST-ambiguous-time crash + crypto tz mislabeling (fixed 2026-08-20, upstream)
1h-interval fetches spanning a US DST fall-back date (e.g. 2025-11-02) used
to crash `price_cache.get_price_data()` with `Cannot infer dst time from
... as there are no repeated times`, and Kairos's own local-fallback fetch
(`kairos/data.py`'s `fetch_price_data_local_fallback`) had the identical bug
independently (fixed in this repo directly — see git history around
"BUG-03"). Root cause on the price_cache side was deeper than the DST edge
case alone: yfinance returns crypto intraday bars in UTC (unlike equities,
which come back already in NY time), and price_cache was blindly
`tz_localize(None)`-ing them — mislabeling every crypto intraday bar by 4-5
hours, every day, not just at the DST boundary; the DST crash was just the
one case where the resulting collision was loud enough to raise instead of
silently mislabeling. Fixed at the source (price_cache commit `72bac58`,
propagated through `phantom_ledger`'s submodule bump `8f2d087` and this
repo's `uv.lock`); price_cache's cache schema bumped to v4 to purge
previously-mislabeled cached sub-daily rows (daily-or-longer cache
untouched). If a 1h/sub-daily fetch or backtest run from before this date
looks off by a few hours, or a symbol you know has data mysteriously
disqualifies, this is why — re-fetch after upgrading.

**Gotcha hit while bumping the dependency**: `uv lock --upgrade-package
price-cache` alone silently did nothing (no lockfile change, no error) —
`price-cache` and `phantom-ledger` are two separate `uv` packages sourced
from the *same* `phantom_ledger.git` repo (one via a `subdirectory=`
param), and upgrading only one left the other's cached git checkout of that
shared repo pinned to the old commit; you must pass
`--upgrade-package price-cache --upgrade-package phantom-ledger` together.
Separately, this repo's home directory (`~/.cache`) is bind-mounted onto
the same NTFS/`fuseblk` drive as the repo itself — `uv`'s git checkout
cache under `~/.cache/uv/git-v0` can get into a state where `uv` fails
mid-checkout with `failed to remove directory ... Directory not empty (os
error 39)`, which `rm -rf`-ing that directory does NOT reliably fix (uv
recreates and hits the same error). Work around it by pointing
`UV_CACHE_DIR` at a real (non-`fuseblk`) filesystem for the lock/sync
commands, e.g. `export UV_CACHE_DIR=/tmp/uv-cache-kairos` before
`uv lock`/`uv sync` when bumping a git-sourced dependency.

### Oracle vs. naive-baseline modes, and the parallel dedup sweep (2026-08-27/28)
`kairos_strategies.py` has three prediction modes, not two:
- **Model** (default): real Kronos forecast via `predict_fn`/`multi_predictor`.
- **Oracle** (`--no-prediction`): replaces the model's distribution with the
  *actual next-bar* OHLCV (`_make_realized_predictions()` in
  `kairos_orchestrator.py`, `config.no_prediction=True`) — a
  perfect-foresight **ceiling**. This is what the `--stage oracle` pipeline
  stage runs; it also drives the production `disabled_strategies` gate via
  `refresh_disabled_strategies()`.
- **Naive baseline** (`--naive-baseline`, implies `--no-prediction`,
  `config.naive_baseline=True`): keeps oracle's real decision (direction +
  relative stop/target %, from its genuine future-peeking distribution)
  completely unchanged — the *decision* is not recomputed. What changes is
  the accounting: entry is re-anchored to the real bar oracle peeked at (by
  the time this trade could exist, that bar has closed for real — it's no
  longer a peek), stop/target recomputed from the same relative offsets
  against that new entry, and the trade resolved only against genuinely
  later bars, walking forward until a real stop/target trigger — or
  excluding the signal if data runs out first, never force-closing it at an
  arbitrary point. Implemented in
  `KairosOrchestrator._compute_shadow_performance_naive()`, mirroring the
  exact terminal-exit-reason contract `kairos_signal_replay.py` already
  established for `BacktestEngine._check_exit` (open-gap-then-intrabar,
  "close" means keep holding not force-exit) — reimplemented as a plain
  pandas loop, no `BacktestEngine`/phantom_trader dependency. A **floor**:
  measures how much of a strategy's edge depends on prediction quality at
  all, with zero future peek anywhere in the decision.

  **This mode went through a real methodology correction on 2026-08-28,
  worth knowing before touching it again.** An earlier implementation
  (`config.use_current_bar`, since removed) tried to answer the same
  question by feeding the distribution-construction step the *current*
  bar as if it were the forecast basis (same bar for both center and
  shape) — this doesn't produce a neutral no-information test, it bakes in
  an active "assume zero drift" assumption that handicaps every directional
  strategy by construction, since the distribution ends up centered exactly
  at the entry price. A full 961-group sweep was run on this flawed version
  before the flaw was caught; that DB data (`stage='naive'` at the time) was
  deleted outright rather than kept as a labeled-flawed artifact. The
  corrected version above (originally prototyped under the name
  "lagged oracle" mid-session, then renamed to take over the `naive`
  identity once confirmed correct) reuses oracle's real decision instead of
  re-deriving one from a self-referential bar, which avoids the same trap.

  Also surfaced while fixing this: `_compute_shadow_performance()` (oracle's
  own evaluator, and the one every `oracle_results` Sharpe/win-rate number
  in this table has ever come from) only checks **one bar ahead**, then
  force-closes at that bar's close if neither stop nor target triggered —
  it does not walk forward multiple bars. Don't assume "the current
  mechanism already handles multi-bar TP/SL" without checking which
  evaluator you mean; the genuinely multi-bar-capable one is
  `BacktestEngine._check_exit`/`kairos_signal_replay.py`'s usage of it, not
  this one.

`run_stage_naive()` in `kairos_pipeline.py` mirrors `run_stage_oracle()` but
writes `stage='naive'` rows to the same `oracle_results` table (it already
has a `stage` column) and **deliberately never calls
`refresh_disabled_strategies()`** — mixing naive-baseline Sharpe into the
production disable gate would incorrectly disable strategies that work fine
with real predictions but (expectedly) go quiet or lose money once
prediction is stripped out entirely. Naive-baseline results are for
analysis only.

**Six independent TP/SL-checking implementations exist in this codebase**
(found while investigating the one-bar issue above) — worth knowing before
adding a seventh or assuming "the backtest engine" means one specific thing:
1. `_compute_shadow_performance()` — 1 bar only, feeds `oracle_results`/`model_results`.
2. `_compute_shadow_performance_naive()` — multi-bar, correct, naive-only.
3. ~~`KairosOrchestrator._manage_positions()`~~ — **removed entirely
   2026-08-28**, see below. Historical only.
4. `BacktestEngine._check_exit`/`_calculate_pnl` (`kairos_backtest.py`) —
   used only by `kairos_signal_replay.py`; multi-bar, no phantom dependency.
5. `MultiHorizonBacktestEngine._check_exit` (`kairos_horizon.py`) —
   **dead code**, imported into `kairos_orchestrator.py` but never
   instantiated anywhere.
6. `PartialExitBacktestEngine`/`_check_leg_exit` (`kairos_execution.py`) —
   **also dead code**; the `execution_plan` metadata that would route here
   was captured from signal metadata but never read anywhere (still true;
   `UnifiedSignal` itself no longer exists, see below).
7. phantom_ledger's `PositionManager.determine_close` — the actual live
   papertrade fill engine (`kairos_papertrade.py`), external submodule,
   unrelated to any of the above.

**The real portfolio simulation was removed entirely (2026-08-28), same day
it was found and fixed for `hold_days` above.** Once `hold_days` was fixed,
Baz asked directly: since nothing persisted reads `_manage_positions`'s
output, why run it at all? Answer, after a full input/output dependency
trace: almost everything was safe to remove, except one real coupling —
`OvernightExposureFilter` (`kairos_backtest.py`) read `context["current_position"]`,
sourced from `self.active_positions`, which **is** shadow-persisted (it's a
registered strategy, unconditionally active). Fixed by giving that one
filter its own self-contained shadow position tracker
(`self._shadow_position: Dict[str, Dict]`, keyed by symbol, set from its
own wrapped strategy's signals) instead of reading real portfolio state —
decoupled from capital/exposure/competitive-selection entirely. Deliberate
simplification: it doesn't model a stop/target hit closing the shadow
position early (there's no real engine left to hit one against) — it only
closes via its own overnight check.

With that one blocker resolved, **removed**: `_manage_positions()`,
`_enter_position()`, `_close_all_positions()`, `_create_unified_signal()`,
the `UnifiedSignal` dataclass, `get_live_signal()` (confirmed zero callers
anywhere — dead before this change too), `export_results()` (also zero
callers, and would have crashed anyway since it referenced keys this change
removes), `self.tracker`/`StrategyPerformanceTracker` instance (its
`get_weight`/`record_trade` calls had no remaining caller once
`_manage_positions` and `get_live_signal` were gone), and the portfolio
allocator / cross-asset-ranking / per-asset "best signal" selection steps in
`_run_day()` (they only ever fed the now-removed entry step; shadow
recording already happens earlier and is untouched by any of them).
`self.capital`/`self.active_positions`/`self.all_trades`/`self.daily_logs`/
`self.all_signals` are gone; `self.equity_curve` survives as a plain list of
dates (day-count only, for the "signals/week" stat in reports — nothing
needs the capital values it used to hold).

**`results["summary"]` (Total Return, Sharpe, Max Drawdown, Win Rate, etc. —
the CLI banner) is now built from `shadow_performance` instead**, via a new
shared helper `_compound_equity_stats()` (module-level in
`kairos_orchestrator.py`) extracted from `backtest_top_strategies()`'s
pre-existing per-strategy pattern (compound each pnl_pct at 10% of capital
per signal). The summary pools **every strategy's every shadow signal**
into one combined list — an approximation that assumes every signal could
have been taken independently with equal sizing, explicitly not a claim
about a single capital-constrained account. `best_strategy`/`worst_strategy`
now come from `shadow_ranked` (same source as `strategy_rankings`) instead
of a real-trade ranking that no longer exists. One cosmetic side effect:
`max_drawdown`'s sign convention now matches `backtest_top_strategies`'s
existing (negative) convention throughout, rather than the old top-level
banner's separate positive-sign convention — purely cosmetic, nothing
persisted or parsed depends on the sign.

**Real, separate blind spot found and fixed during this change**:
`kairos_signals.py`'s own `_build_context()` — a third, independent
context-building function (used by the real live-signal-generation
pipeline, not `run_backtest()`) — read `orchestrator.capital` as a live
attribute. Removing `self.capital` broke it (silently, inside a try/except,
producing wrong cache/predict behavior rather than a clean crash — caught by
20 `test_signals_report.py` failures). Fixed the same way as everywhere
else: removed the dead `"capital"`/`"current_position"` context keys from
`_build_context()` too. Lesson: `context["capital"]`/`context["current_position"]`
had **three** independent producers across the codebase
(`_run_day()`, `get_live_signal()`, `kairos_signals._build_context()`) —
grepping only inside `kairos_orchestrator.py` and the strategy files misses
call sites like this one; grep the whole repo for `orchestrator\.<attr>`
patterns too when removing orchestrator state.

**Verified nothing persisted was affected**: full test suite green (1718
passed) after the fix above; a fresh `run_stage_oracle()` call post-refactor
produced byte-identical Sharpe/win_rate/signal_count values to the same
group's pre-refactor numbers (`_compute_shadow_performance()` itself was
never touched — only what happens downstream of it).

**`oracle_results.version` column (added 2026-08-28).** Purely
administrative — `kairos_pipeline.git_commit_hash()` (short hash, cached
per-process) is written on every `insert_oracle_row()` call, so a future
session can tell which code version produced a given row without guessing
from timestamps. Existing rows predating this column are `NULL` (no
retroactive backfill — genuinely unknown). Excluded from
`_get_metric_columns()`'s auto-discovered metric list (it's not a metric).
`model_results` doesn't have this column yet — same rationale would apply
if it's ever wanted there.

**The oracle dedup sweep is now a parallel process pool, not sequential.**
`scripts/run_oracle_dedup.py` (committed; supersedes an original sequential
scratchpad script of the same name from 2026-08-26) runs
`select_deduped_groups()` through a `ProcessPoolExecutor` instead of one
`subprocess.run()` at a time. Root cause of the old script's idle cores
(observed live: one core pegged, others 5-40%, hopping across cores every
few seconds): the per-day/per-strategy backtest loop is mostly
single-threaded, GIL-bound Python, not BLAS matrix math — the occasional
`ps` reading of ~4 cores' worth of cumulative CPU was brief vectorized
bursts blended into the average, not sustained parallelism. Real
multiprocessing (separate OS processes, separate GILs) was the fix. Each
worker subprocess is pinned to 1 BLAS thread
(`OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/
`NUMEXPR_NUM_THREADS=1`) so N workers approximate N busy cores instead of
oversubscribing. Measured on this box (8 physical / 16 logical cores): 4
workers gave a 3.78x speedup over sequential on matched 4-asset groups,
close to linear — `--workers` defaults to 8 (physical core count; SMT
doesn't reliably double GIL-bound throughput). Supports `--stage
oracle|naive` (default `oracle`) to sweep either mode across the same
deduped group list; same resume/skip semantics as before (checks
`oracle_results` for an existing `(assets, interval, backtest_period,
stage)` row before running a group).

**Multi-session caution**: this box regularly runs more than one Claude
Code session at once (see `docs/handoff-*.md` for the current cast). A
`kairos_strategies.py` subprocess spawned by *your* test can look
identical in `ps` to one spawned by a sibling session's live sweep — same
command shape, same args style, no attribution. Before killing anything
that looks like an orphan, verify it's actually yours (e.g. by asset list
match against what you just launched, or timestamp), not just "looks like
a leftover." A live sweep's own group was killed by mistake this way on
2026-08-27 — the sweep's per-group error handling absorbed it as a `[FAIL]`
and moved on, but the group still needed a manual one-off re-run to backfill.

### Eight strategy names are the same strategy (found 2026-08-29)

`trend_following` plus seven "filter" strategies — `cds_spread_filter`,
`cot_positioning_filter`, `dark_pool_filter`, `fractal_dimension`,
`gaussian_process`, `insider_cluster`, `onchain_flow_filter` — all produce
**byte-identical** results. Each of the seven wraps `TrendFollowingStrategy()`
(registered that way in `kairos_orchestrator.py`) and gates on a context key
the pipeline never supplies (`cds_spread_change`, `dark_pool_sentiment`,
`cot_net_position`, …). Every `context.get(key, 0.0)` returns the `0.0`
default, so every gate condition is false and each filter degrades to an
unmodified pass-through, differing only in the name stamped on the signal.
Confirmed in code, not inferred from the numbers.

**Consequence for any corpus-wide statistic: one behaviour votes eight
times.** Collapsing the aliases moved the whitepaper's oracle median from
+0.32 to +2.20, because seven redundant copies of an unprofitable strategy
sat near the middle of the distribution. Counts of *profitable* strategies
were unaffected (all eight are negative in every regime), but medians,
quartiles and denominators all were. Any new per-strategy aggregation should
collapse these to `trend_following` alone — see `docs/papers/build_paper.py`'s
`ALIAS` set and `analyze_by_market3.py` for the pattern.

Four further pairs coincide on most-but-not-all groups and are **not** exact
aliases, so they are deliberately left uncollapsed — but check them before
trusting a fine-grained count: `expected_value`/`vol_target_sizer` (445
groups), `range_trading`/`rqa_determinism` (302),
`dynamic_bracket`/`inverse_variance` (262), `amount_flow`/`predicted_vwap`
(70). Detect aliasing generally by grouping strategies on their full vector
of per-group `(sharpe, signal_count)` and looking for exact duplicates.

### Non-finite bars are skipped, not fatal (fixed 2026-08-29)

`KairosDistribution.from_bar()` seeds its RNG with
`int(abs(actual_close) * 1000)`, so a NaN close raised
`ValueError: cannot convert float NaN to integer` and aborted the **entire
group's** backtest. Previous handoffs described this as an
"international-symbol bug" — **that diagnosis was wrong**. A single missing
bar anywhere in any one symbol's history is enough; the case that finally
pinned it was `P`, with exactly one NaN close in 484 bars. It was
deterministic (the bad bar is the "next bar" for exactly one date), which is
why the same groups failed every sweep.

`_make_realized_predictions()` now skips that symbol for that date. The guard
lives there rather than in `from_bar()` because that is `from_bar`'s only
caller and skipping is the caller's decision; it also covers `current_price`
(whose non-finite case is quieter and worse — it poisons every sampled close
into NaN rather than raising) and `inf` (which overflows the same cast).
**Behavioural consequence:** a symbol with gappy data now silently
contributes fewer signals rather than failing loudly, so a lower-than-expected
`signal_count` may mean skipped bars. It warns once per symbol
(`[warn] <SYM>: non-finite OHLC bar, skipping affected dates`) — grep run logs
for that before concluding a strategy went quiet. Tests in
`tests/unit/test_realized_predictions_nonfinite.py`.

### Per-(model, instrument class) stats: `strategy_class_stats`

Sweeps record per-strategy stats **twice**: the long-standing corpus row (one per
strategy per group, in `oracle_results`/`model_results`) and, since 2026-08-29, a
per-(strategy, asset class) row in `strategy_class_stats`. Motivation is in
`docs/papers/where_strategies_travel.html` — strategy quality is strongly
class-dependent (oracle median +2.40 on equities vs −4.78 on crypto, and some
strategies reverse sign between classes), so one corpus number averages away the
thing you would select on.

**It is a separate table on purpose.** Adding an `asset_class` column to the
results tables would change their grain to (run, strategy, class), and seven
consumers rely on one row per (run, strategy) — `refresh_disabled_strategies()`
would raise `IntegrityError` outright (plain INSERT, PK `(interval, assets,
strategy_name)`), while `run_stage_rebuild_disabled`, `build_viability_report`,
`select_finetune_candidate`, `compare_finetuned_vs_base`, `_get_metric_columns`
and `docs/papers/*.py` would each silently pick one arbitrary class per strategy.
Keeping the results tables untouched makes this purely additive.

**Never reconstruct a corpus figure from per-class rows.** Sharpe is a ratio and
does not recombine across classes — `signal_count` sums and `win_rate` /
`avg_pnl_per_trade` are per-trade means that would recombine exactly, but Sharpe
would not, and it is the number everything reads. The corpus Sharpe stays what it
always was: a true value over the group's pooled `pnl_list` in the results tables.
Within a class, Sharpe is likewise exact, computed from that class's own pooled
list. Read one or the other, never a weighted blend of the per-class rows.
`kairos_pipeline.strategy_class_stats(conn, stage=..., asset_class=...)` enforces
this — `asset_class=None` reads the corpus table, and a class cell below
`CLASS_STATS_MIN_SIGNALS` (30, uncalibrated) falls back to corpus with
`source="corpus"` on the returned dict.

**Attribution is exact for new sweeps, approximate for backfilled rows.**
`_compute_shadow_performance{,_naive}` attribute each signal to its own symbol's
class (the symbol is in `_shadow_signals`; it used to be discarded), so a mixed
group splits correctly. `scripts/backfill_class_stats.py` could not do that for
the 111,657 historical rows — the per-symbol breakdown was gone before they were
persisted — so it derives class from group composition and marks genuinely mixed
groups `'mixed'` (1,503 rows, 1.3%), invisible to per-class reads.

**The invariant that catches attribution bugs:** per-class `signal_count`s must sum
to the corpus `signal_count` for the same (run, strategy). Sharpe will not match
and must never be asserted to.

**Nothing consumes these yet** — that is deliberate. See
`docs/tickets/per-class-stats-wiring.md` for what phase 2 must wire, with call
sites traced.

### Four classifiers, none interchangeable

Do not unify these and do not join on `asset_class` across tables:

| Function | Taxonomy | Grain | Used for |
|---|---|---|---|
| `kairos_backtest.asset_class_of_symbol()` | 3-way `equity｜crypto｜fx_commodity`, suffix-based | symbol | `strategy_class_stats` only |
| `kairos_strategies.asset_class_for()` | 5-way — `fx` and `commodity` **separate**, plus `mixed` | group, majority vote | live `_DISABLED_BY_CLASS` fallback |
| `kairos_pipeline.asset_class_of()` | 3-way, membership lookup in `CANDIDATE_UNIVERSE` | symbol | universe screening |
| `kairos_margin.classify_symbol()` | 7-way margin schedule (`config/margin_ibkr.yaml`) | symbol | leverage/margin math |

The suffix classifier exists because the membership one returns `unknown` for
anything unscreened — which is why 17 real FX-pair groups were unclassifiable in
the market-segmentation analysis.

**`asset_class_for()` is load-bearing and was deliberately left alone.**
`_DISABLED_BY_CLASS` (`kairos_strategies.py:906`) is keyed `("1d","fx")` and
`("1d","commodity")` separately, and `resolve_disabled_strategies()` consults it
for any group with no oracle-tested DB profile — the live path
(`kairos_signals.py:820`). Collapsing fx+commodity into `fx_commodity` there would
make every one of those keys miss, silently returning an **empty disabled set** and
letting strategies as bad as `volume_fade` (−150 mean Sharpe on crypto) run
unfiltered. If you ever do reconcile the taxonomies, update those keys in the same
commit.

### Research papers live in `docs/papers/`

Two published research documents, each generated from the DB rather than
hand-written, with their generators and input data checked in beside them:

| File | What |
|------|------|
| `prediction_premium.html` | *The Prediction Premium* — naive vs base vs oracle, paired on a matched group set |
| `where_strategies_travel.html` | *Where Strategies Travel* — same regimes segmented by asset class and listing venue |
| `build_paper.py` / `paper_table.json` | generator + data for the first |
| `build_market_report.py` / `market_analysis3.json` | generator + data for the second |
| `analyze_by_market3.py` | regenerates `market_analysis3.json` from `pipeline_results.db` |
| `audit_paper.py` | re-derives the first paper's figures straight from the DB and cross-checks the published HTML |

Both generators read every figure from JSON instead of transcribed constants,
specifically so a page cannot drift from the data — earlier hand-kept
constants printed a median of `+0.160` where the data said `+0.159`. Re-run
the analysis and the pages regenerate.

**Two flavours, one source.** Run a generator **from `docs/papers/`** and it
writes a standalone doctype'd file there (for opening from disk — `file://`
has no Content-Type header, so without `<meta charset>` the em-dashes and ρ
garble). Run it **from a scratchpad** and it writes a head-less
artifact-publish source there *and* refreshes the repo copy. Publish the
scratchpad copy — the Artifact tool rejects a page carrying its own doctype.

The §8 "Data provenance" section of each paper records the exact tables,
columns, run ids and SQL behind every figure; that is the reference for
"where does the data for phase X live," not this file.

## Test suite

Tests live in `tests/unit/` and require no GPU or model download.

| File | What it covers |
|------|----------------|
| `test_kairos_distribution.py` | `KairosDistribution`: entropy, stats, EV, CDF, Kelly |
| `test_backtest_engine.py` | `backtest()` and `compute_metrics()` functions |
| `test_strategy_signals.py` | Individual strategy `generate_signal()` logic |
| `test_filters.py` | `KurtosisFilterStrategy` and `_apply_meta_filters` |

`tests/conftest.py` adds `strategy/` to `sys.path` for all test files.

## OrchestratorConfig defaults (after calibration)

| Parameter | Default | Why |
|-----------|---------|-----|
| `entropy_threshold` | 3.0 | Matches max Shannon entropy for 20 bins (ln 20 ≈ 3.0) |
| `kurtosis_max` | 10.0 | Discrete token samples routinely exceed 3.0 |
| `kurtosis_action` | `"block"` | Skip high-kurtosis days entirely |
| `min_volume_percentile` | 10.0 | Model volume predictions are mean-reverting; 30 was too strict |
| `debug_filters` | `False` | Set True to print entropy/kurtosis per asset per day |

These are the `1d` defaults, read directly off the `OrchestratorConfig`
dataclass. As of E12-S02 (2026-08-20), `1h` also has an explicit entry in
`kairos_orchestrator.py`'s `_FILTER_PRESETS_BY_INTERVAL` (consumed via
`OrchestratorConfig.for_interval(interval, ...)`) — a live `debug_filters=True`
sweep (n=4579 filter evaluations, CL=F/NG=F/SI=F/ZW=F/MKR-USD, 1h bars) found
entropy never exceeds ~2.9 (same ln(20)≈3.0 ceiling as 1d) and kurtosis only
exceeds 10 in 0.66% of samples (p99=8.6) — statistically the same shape as
1d, so `1h` uses the identical values (`entropy_threshold=3.0`,
`kurtosis_max=10.0`, `min_volume_percentile=10.0`), verified rather than
assumed. The sweep sample was thin and fx_commodity/single-crypto-skewed
(most crypto still fails 1h universe screening on a separate `$vol=0.0`
issue, see `docs/todo.md`'s BUG-04 entry) — worth re-running once that's
fixed and a broader crypto sample is available. See
`kairos_orchestrator.py`'s `_FILTER_PRESETS_BY_INTERVAL["1h"]` comment for
the full percentile breakdown.
