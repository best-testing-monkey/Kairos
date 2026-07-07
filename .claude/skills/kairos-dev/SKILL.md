---
name: kairos-dev
description: Use for any Kairos development, debugging, or backtesting task in this repo (strategy/, model/, kairos/, tests/) — architecture map, canonical run/test commands, smoke test, and known environment pitfalls, so you don't re-read the big strategy modules or regenerate the smoke heredoc.
---

# Kairos dev reference

`strategy/` is NOT a package (no `__init__.py`) — modules import each other by
bare name, so any script/test must `sys.path.insert(0, ".../strategy")` first
(see `tests/conftest.py`). `kairos/` (package) wraps `price_cache` config.
See also repo-root `CLAUDE.md` for calibration gotchas (entropy/kurtosis
thresholds, percentile key formats) — not duplicated here.

## Verifying allocator integration (E2-S12+)
`KairosOrchestrator(...).registry.register_allocator(alloc)` before
`.run_backtest(...)` routes sizing through `apply_allocator` each day.
Smoke-style probe: reuse `scripts/smoke.py` helpers (exec the file body,
supply `__file__` in globals), wrap `alloc.allocate` with a counter, assert
it fires once per backtest day. Zero-size trades appear in the synthetic
smoke fixture even without an allocator - pre-existing, not a regression.

## Architecture map (read this instead of re-reading the files)

**`strategy/kairos_backtest.py`** (1967 lines) — core data model, no orchestration:
- `KairosSettings.configure(args)` — applies CLI args as module globals.
- `Direction` / `Regime` (Enum), `Signal`, `Trade` (dataclasses).
- `fast_concat(predictions) -> DataFrame`, `distribution_for(predictions) -> KairosDistribution`.
- `KairosDistribution(predictions)` — stats over sampled prediction paths: `.entropy()`, `.cdf()`, `.pdf()`,
  `.predicted_sharpe()`, `.kelly_fraction(entry, target, stop)`, `.expected_value(...)`,
  `.overlap_coefficient(other)`, `.is_bimodal()`, `.coefficient_of_variation()`, `.predicted_range()`,
  classmethod `.from_bar(bar, n_samples=100)`.
- `KairosPredictor(predict_fn).predict(history) -> KairosDistribution`.
- `Strategy` (base) — subclasses implement `generate_signal(dist, current_price, history, context)`.
  Built-ins here: `PercentileEntryStrategy`, `DynamicBracketStrategy`, `SkewStrategy`,
  `RangeTradingStrategy`, `TrendFollowingStrategy`, `VolatilityArbStrategy`, `HighLowStrategy`,
  `OpenGapStrategy`, `FadeExtremeStrategy`, `MomentumContinuationStrategy`, `ExpectedValueStrategy`.
  **All `generate_signal` implementations must return a `Signal` or `None`** — never a plain dict
  (breaks `LiquidityFilterStrategy.metadata` access).

**`strategy/kairos_orchestrator.py`** (1337 lines) — the entry point, most-read file (68x historically):
- `OrchestratorConfig` — dataclass of every tunable knob (fees, meta-filter thresholds, cross-asset,
  partial exits, `disabled_strategies` set of ~19 shadow-tested-unprofitable strategy names).
- `StrategyRegistry.build_all(config) -> List[Strategy]` — builds all 42 strategies.
- `UnifiedSignal` — cross-strategy signal wrapper.
- `KairosOrchestrator(predict_fn, assets=None, config=None, batch_predict_fn=None, **kwargs)`:
  - `.run_backtest(data_dict: Dict[str, DataFrame], lookback=200) -> Dict` — full walk-forward backtest;
    raises `ValueError("No common dates found across assets after lookback")` if histories shorter than
    `lookback` don't overlap enough — pass enough bars or lower `lookback`.
  - `.run_single_asset(df, lookback=200) -> Dict`, `.get_live_signal(histories) -> Optional[UnifiedSignal]`.
  - Internals: `_make_realized_predictions`, `_run_day`, `_apply_meta_filters`, `_manage_positions`,
    `_enter_position`, `_calculate_pnl`, `_close_all_positions`, `backtest_top_strategies`,
    `_compute_shadow_performance`, `_build_results`.
  - Module functions: `print_results(results, ...)`, `export_results(results, filepath)`.

**`strategy/kairos_strategies.py`** (728 lines) — CLI demo / data fetch:
- `fetch_data_raw(symbol, lookback, pred_len=0, min_bars=None) -> DataFrame`, `fetch_data(...)`.
- `is_24_7_crypto_symbol`, `calendar_days_for_bars` — exchange-calendar-aware bar math.
- `_ensure_model_loaded(model_path=None, tokenizer_path=None)` — lazy Kronos load, sets TF32 flags on CUDA.
- `run_model(...)`, `predict_all_batch(assets: dict) -> dict` — batched Kronos inference.
- `backtest(predicted_close, actual_close, ...)`, `compute_metrics(equity, initial_capital, trades)`.
- `parse_signals_config`, `compute_signals`, `resolve_disabled_strategies(interval, assets)`.
- `__main__` block (line ~643) defines the CLI (see Commands below); `PRED_SAMPLES=100`,
  `DEMO_LOOKBACK=300` — do not shortcut these, they're calibrated.

**`model/kronos.py`** (706 lines) — the forecasting model itself:
- `KronosTokenizer(nn.Module, PyTorchModelHubMixin)` — VQ-style encode/decode (`.encode`, `.decode`, `.indices_to_bits`).
- `Kronos(nn.Module, PyTorchModelHubMixin)` — transformer body: `.forward`, `.decode_s1`, `.decode_s2`.
- `auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, ..., sample_count=5)`
  — the sampling loop; key perf trick: encode/decode_s1 once at original batch size, expand only for
  the stochastic sampling step (~0.3s/iter GPU vs 89s naive).
- `KronosPredictor(model, tokenizer, device=None, max_context=512, clip=5)`:
  `.predict(df, x_timestamp, y_timestamp, pred_len, ...)`, `.predict_batch(df_list, ...)`.

**Other strategy modules** (each defines `Strategy` subclasses, same `generate_signal` contract):
`kairos_execution.py` (partial exits/pyramiding/liquidity, `PathExecutionPlanner`, `VolumeAnalyzer`),
`kairos_path.py` (`KairosPathExtractor`, `PathProfile`, path-shape strategies),
`kairos_meta.py` (cross-asset + regime/tail strategies, `StrategyPerformanceTracker`),
`kairos_pipeline.py` (asset-discovery pipeline stages: universe/correlation/oracle/model → sqlite,
see `strategy/PIPELINE.md`), `kairos_crypto.py` / `kairos_forex.py` / `kairos_stocks.py` /
`kairos_universal.py` (asset-class-specific strategy sets), `kairos_horizon.py` (multi-horizon holds).

## Canonical commands

```bash
# Fast iteration backtest demo, no live model download/GPU wait, ~26x used historically
uv run ./strategy/kairos_strategies.py --no-prediction --pred_samples 100 [--interval 1h]

# Full unit suite (no GPU/model needed) — 208 tests, ~11s
uv run --with pytest python -m pytest tests/unit/ -q

# Single test file
uv run --with pytest python -m pytest tests/unit/test_kairos_distribution.py -v

# Smoke test: minimal KairosOrchestrator + dummy predict_fn + tiny synthetic backtest.
# Encapsulates what used to be a hand-typed heredoc (~22x). Verified working (208-passed baseline, ~5s).
uv run --with pytest python scripts/smoke.py

# Asset-discovery pipeline (see strategy/PIPELINE.md for full stage/flag list)
uv run ./strategy/kairos_pipeline.py --stage universe
```

Other `kairos_strategies.py` flags (verified in the `__main__` argparse block):
`--model PATH`, `--tokenizer PATH`, `--symbol SYM`, `--assets SYM [SYM ...]`,
`--backtest_period PERIOD`, `--lookback N`, `--initial_capital N`, `--export_json PATH`.

## Environment gotchas

- **hatchling editable-build errors**: this repo is not itself installed as an editable package in
  the sense of `strategy/` — only `kairos/` is `[tool.hatch.build.targets.wheel] packages = ["kairos"]`.
  If `uv run` fails trying to build the project, check you're not accidentally importing `strategy.*`
  as a package; it's added via `sys.path`, not installed.
- **pandas_ta vs numpy/pandas conflicts**: `pandas_ta` is NOT a core dependency (grep confirms it only
  appears in `examples/run_backtest_kairos_html.py`) — the core dep is `ta` (`ta>=0.10` in
  `pyproject.toml`). If a session pulls in `pandas_ta`, expect numpy/pandas pin conflicts; prefer the
  `ta` package already used by core code, or isolate `pandas_ta` example runs with `uv run --with pandas_ta`.
- **price_cache git dependency**: pinned via `price-cache @ git+https://github.com/best-testing-monkey/price_cache.git`
  (see `uv.lock` for the resolved commit). A local patched checkout lives at `/tmp/price_cache_fix` —
  if `uv run` picks a stale/broken commit, check that fix tree first before re-debugging from scratch.
- **price_cache DST / tz_localize quirk**: `price_cache/_cache.py` normalizes timestamps with
  `.dt.tz_localize(None)` / `.normalize().tz_localize(None)` in several places (history, date columns).
  Naive-vs-aware mismatches around DST transitions are the recurring symptom — if you see
  `tz_localize` errors, check `_cache.py` around lines 834–939 and 1425/1623/1936 (in the
  `/tmp/price_cache_fix` checkout) rather than patching Kairos-side code.
- **torch.compile unsupported**: Python 3.13 + `torch.compile` raises `Dynamo is not supported`; use
  TF32 flags instead (already set in `_ensure_model_loaded()`).
