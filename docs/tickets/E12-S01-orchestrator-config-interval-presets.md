# E12-S01 — Add an interval-keyed `OrchestratorConfig` preset mechanism

**Goal:** `OrchestratorConfig`'s meta-filter thresholds (`entropy_threshold`, `kurtosis_max`, `min_volume_percentile`) are fixed dataclass defaults with no per-interval variation point. Add a preset mechanism (mirroring `_DISABLED_BY_CLASS`'s `(interval, ...)` keying pattern already used for disabled strategies) so a future calibration pass (E12-S02) has somewhere to put `1h`-specific numbers, without changing `1d` behavior at all in this story.

**Context:**
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §2 (E3/E12 section).
- `strategy/kairos_orchestrator.py`, `class OrchestratorConfig` (search `class OrchestratorConfig`, ~line 239) — a plain `@dataclass` with `entropy_threshold: float = 3.0`, `kurtosis_max: float = 10.0`, `kurtosis_action: str = "block"`, `min_volume_percentile: float = 10.0` among its fields (~line 248-254). These four numbers are exactly the CLAUDE.md-documented "OrchestratorConfig defaults (after calibration)" table — treat them as the canonical `1d` values; do not change them.
- `strategy/kairos_strategies.py`, `_DISABLED_BY_CLASS: dict` (search `_DISABLED_BY_CLASS`, ~line 875) is the existing pattern to mirror: a module-level dict keyed by `(interval, ...)`, with a documented fallback to "nothing" when the key is absent — do the same shape here (a dict keyed by bare `interval` this time, not `(interval, asset_class)`, since these are global filter thresholds, not per-asset-class).
- Only 2 real call sites need wiring in this story (found via `grep -n "OrchestratorConfig(" strategy/*.py`):
  - `strategy/kairos_signals.py`, `_run_group` (~line 823): `config = OrchestratorConfig(disabled_strategies=disabled)` — `interval` is already a parameter of `_run_group` itself, right there in scope.
  - `strategy/kairos_strategies.py`, the `__main__` CLI block (~line 1033): `config = OrchestratorConfig(initial_capital=..., ...)` — `KairosSettings.interval` is already set by this point (`KairosSettings.configure(args)` runs earlier in the same block).
  - Also update `strategy/kairos_orchestrator.py`'s `KairosOrchestrator.__init__` fallback (~line 735): `self.config = config or OrchestratorConfig()` → `self.config = config or OrchestratorConfig.for_interval(KairosSettings.interval)`. `KairosSettings` is already imported into this file (see the `from kairos_backtest import (... KairosSettings ...)` block near the top).
  - Do NOT touch `kairos_orchestrator.py`'s `__main__` block example text (~line 1605) — that's inside a `print("""...""")` documentation string, not executable code.

**Acceptance criteria:**
- [ ] New module-level dict in `kairos_orchestrator.py`, near `OrchestratorConfig`: `_FILTER_PRESETS_BY_INTERVAL: dict[str, dict] = {}` — **empty for now**. Leaving it empty means every interval (including `"1h"`) falls back to the dataclass's own hardcoded defaults until E12-S02 populates real calibrated numbers; this story is infrastructure only, not calibration.
- [ ] New classmethod on `OrchestratorConfig`: `for_interval(cls, interval: str, **overrides) -> "OrchestratorConfig"` — looks up `_FILTER_PRESETS_BY_INTERVAL.get(interval, {})`, merges it with `overrides` (overrides win on conflict), and constructs `cls(**merged)`. Docstring should explain: presets fill in interval-specific defaults for entries in `_FILTER_PRESETS_BY_INTERVAL`; any interval not in that dict silently falls back to the class's own dataclass defaults (today's `1d`-calibrated numbers) — this is intentional so an uncalibrated interval doesn't crash, it just inherits `1d`'s thresholds until someone calibrates it.
- [ ] `kairos_signals.py:823` becomes `config = OrchestratorConfig.for_interval(interval, disabled_strategies=disabled)`.
- [ ] `kairos_strategies.py`'s `__main__` block's `OrchestratorConfig(...)` call becomes `OrchestratorConfig.for_interval(KairosSettings.interval, ...)` with the same existing kwargs.
- [ ] `kairos_orchestrator.py:735` becomes `self.config = config or OrchestratorConfig.for_interval(KairosSettings.interval)`.
- [ ] For `interval="1d"` (or any interval not in `_FILTER_PRESETS_BY_INTERVAL`, which is all of them right now): `OrchestratorConfig.for_interval("1d")` produces field values identical to `OrchestratorConfig()` — confirm with a unit test asserting `OrchestratorConfig.for_interval("1d").entropy_threshold == OrchestratorConfig().entropy_threshold == 3.0` (and same for `kurtosis_max`, `min_volume_percentile`).
- [ ] Unit test: `_FILTER_PRESETS_BY_INTERVAL["__test_only__"] = {"entropy_threshold": 1.5}` (monkeypatched in the test, not a real interval) then `OrchestratorConfig.for_interval("__test_only__").entropy_threshold == 1.5` while other fields stay at dataclass defaults — proves the merge/override logic works before any real preset exists.
- [ ] Unit test: `OrchestratorConfig.for_interval("1d", disabled_strategies={"foo"}).disabled_strategies == {"foo"}` — proves explicit overrides still work alongside presets.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped to `strategy/kairos_orchestrator.py`, `strategy/kairos_signals.py`, `strategy/kairos_strategies.py`).
- [ ] New tests pass; full suite green (`uv run --with pytest python -m pytest tests/unit/ -q`) — pay particular attention to any existing test that constructs `OrchestratorConfig()` directly and asserts specific threshold values; those must be unaffected since this story doesn't change any default.
- [ ] Changes committed and `docs/todo.md` E12-S01 item checked off.
