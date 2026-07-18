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

### PRED_SAMPLES and DEMO_LOOKBACK
`PRED_SAMPLES = 100` and `DEMO_LOOKBACK = 300` in the `__main__` block of
`strategy/kairos_strategies.py` are hard constraints. Do not reduce them as a
performance shortcut — change the code instead.

### Backtesting performance (GPU)
The key optimization in `auto_regressive_inference`: run `tokenizer.encode` and
`model.decode_s1` once at the original batch size (3 assets), then expand to
`batch_orig × sample_count` only for the stochastic sampling step. This gives
~0.3s/iteration on GPU vs. 89s before.

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
