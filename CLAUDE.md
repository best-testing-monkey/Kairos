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
shared disk cache — so each model loads exactly once for the whole run. The
date-major `run()` loop that follows then finds every `(symbol, bar, model)`
prediction already cached and never reloads at all
(`kairos_strategies.predict_all_batch` defers `_materialize_model` until
*after* the shared-cache lookup, and skips it entirely on a full hit).

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
to be fixed before a full 6-month run finished cleanly (RSS stable ~5.28GB):

- **`is_batch_cached()` was mutating the shared LRU.** It called
  `PredictionCache.get()` purely for a boolean check, but `get()` always
  promotes a disk hit into `_mem` as a side effect — so every prewarm
  check-pass lookup was silently growing the in-memory cache the same as a
  real prediction would. Fixed with `PredictionCache.has()`, an
  existence-only check that never touches `_mem`. If you ever need a
  read-only cache-hit check elsewhere, use `.has()`, not `.get()`.
- **`_prediction_cache` (the per-process dict in `kairos_strategies.py`,
  distinct from `kairos_predcache`'s disk-backed `PredictionCache`) is only
  cleared on a model switch** (`_prepare_model_switch`). The base sweep
  processes one model for its entire ~20k-entry pass with no switch, so
  this dict grew unbounded for the whole sweep. Fixed with a hard 5000-entry
  cap (`_PREDICTION_CACHE_MAX_ENTRIES`) in `_prediction_cache_put()` —
  clears the whole dict on overflow rather than evicting individually.
- **GC starvation during long same-model sweeps.** `gc.collect()` previously
  only fired inside `_materialize_model()` on a model switch. CPython's
  generational GC thresholds are allocation-*count* based, not memory-size
  based, so large-but-few pandas DataFrames can sit as uncollected garbage
  for a long time with nothing triggering a collection. Fixed with a
  periodic `gc.collect()` every `_PREWARM_GC_INTERVAL` (500) iterations in
  all four prewarm check/load loops (base and finetuned).
- **`sqlite3.OperationalError: unable to open database file`** and
  **`OSError: [Errno 24] Too many open files`** both crashed live runs.
  Mitigated (not root-caused with full certainty — never deterministically
  reproduced outside the long run) with `kairos_signals._connect_with_retry()`
  (3 attempts, 1s/2s backoff, used by both `kairos_signals.run()` and
  `prewarm_prediction_cache()`) and `kairos_papertrade._raise_fd_limit()`
  (raises the process's soft `RLIMIT_NOFILE` to its hard cap, called as the
  first line of `main()`, best-effort/non-fatal on failure).
- **Prewarm speed**: the check pass now stops at the first cache miss per
  unit (base, or a finetuned group) instead of exhaustively checking every
  remaining entry — one miss is already enough to know a load is needed.
  When a unit's check pass finds zero misses, the load pass is skipped
  entirely (logged to console: `"Prewarm load: <unit> skipped -- check pass
  found no cache misses"`). When a load pass does run, it always covers the
  *full* cross product for that unit, not just what the check pass happened
  to see before stopping early.

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
