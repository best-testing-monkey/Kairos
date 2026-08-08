# Appendix A: Implementation Standards

- **generate_signal() must return a Signal dataclass or None**: All `generate_signal()` implementations must return a `Signal` dataclass (from `kairos_backtest.py`) or `None`, never a plain `dict`. Returning a dict silently breaks the `LiquidityFilterStrategy` wrapper which accesses `.metadata`.

- **Percentile stats dict keys use int format**: Percentile stats dict keys are formatted as `f"pct_{int(x)}"` (int, no decimal). Always cast float percentile params to `int` before formatting into the key, e.g., `f"pct_{int(self.stop_pct)}"`.

- **Hard dependencies are numpy, scipy, pandas only**: Only `numpy`, `scipy`, `pandas` are hard dependencies in `strategy/` code. `sklearn` may be used optionally with a manual fallback (see `shrunk_covariance` in kairos_portfolio.py). Never import `arch`, `statsmodels`, `cvxpy`, or `stumpy` in strategy code — each has a small pure-numpy/scipy implementation specified in the design doc, validated against reference fixtures generated once from the real library and checked in as CSV.

- **Stateful strategies must cache fits on fixed cadence**: Anything that fits a model (GARCH, GBM, LPPLS, meta-labeler, GA allocator) must cache its fit and refit only on a fixed cadence (e.g., weekly) — never per-bar. This protects the ~0.42s/iteration GPU backtest budget; fitting must be CPU-only and off the inference hot path.

- **Stateful strategies/allocators must expose reset() method**: Stateful strategies/allocators (universal portfolio, meta-labeler, changepoint guard) keep state on the instance (following the `StrategyPerformanceTracker` pattern) and must expose a `reset()` method so walk-forward folds start clean.

- **strategy/ modules import by bare module name**: `strategy/` modules import each other by bare module name (no package prefix) since `strategy/` has no `__init__.py` and is not a Python package; `tests/conftest.py` adds `strategy/` to `sys.path` for all test files, so tests do not need to do this themselves.

- **Run unit tests with uv**: Unit tests are run with: `uv run --with pytest python -m pytest tests/unit/<file> -q`.

- **Commit convention with trailers**: Commit each story individually with a message of the form `"<story-id>: <title>"`, and the commit message must end with these two trailer lines exactly:
  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_0178edBvkUKM6QXCicY2erxQ
  ```

- **Update todo.md after commit**: After committing a story, check off its line in `docs/todo.md` (change `- [ ]` to `- [x]`).

- **Reuse thresholds from OrchestratorConfig**: Reuse the existing `OrchestratorConfig` entropy/kurtosis thresholds (`entropy_threshold` default 3.0, `kurtosis_max` default 10.0) wherever a strategy needs entropy or kurtosis gating — never introduce a parallel/duplicate threshold constant.
