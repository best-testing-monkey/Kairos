# E12-S02 — Calibrate `1h` filter thresholds via a live `debug_filters=True` sweep

**Goal:** Populate `_FILTER_PRESETS_BY_INTERVAL["1h"]` (added as an empty dict in E12-S01) with real numbers derived from actual `1h` prediction data, not guessed.

**⚠️ This story requires a GPU and live model inference. Do NOT run it via an unattended/automated `/run-stories` loop — it needs a working GPU (or `KAIROS_ALLOW_CPU=1`, much slower), can take a while, and the resulting numbers need a human (or at least a careful review pass) to sanity-check before they're trusted as the new `1h` defaults. Execute manually or under direct supervision.**

**Context:**
- Depends on E12-S01 (the preset mechanism must exist first) and E10/E11 (universe + correlation for `1h` should exist so there's a real `1h` group to test against).
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §2 (E3/E12 section) and the CLAUDE.md "OrchestratorConfig defaults (after calibration)" table — that table's `1d` numbers (`entropy_threshold=3.0`, `kurtosis_max=10.0`, `min_volume_percentile=10.0`) were themselves derived by exactly this kind of sweep; use the same method for `1h`.
- `OrchestratorConfig.debug_filters: bool = False` (added to the dataclass already) — set `True` to print per-asset entropy/kurtosis/volume-percentile diagnostics per day, per CLAUDE.md's own note on this flag.
- `KairosDistribution.entropy()` computes Shannon entropy (PMF-based, range `0`–`ln(20)≈3.0` for 20 bins) — CLAUDE.md warns explicitly NOT to revert to `density=True` in `np.histogram`, which would produce a completely different (differential entropy) scale. Whatever `1h` sweep numbers come out of this story must be on the SAME Shannon-entropy scale as the `1d` numbers — if the sweep output looks like it's in the `12`-`14` range instead of `0`-`3`, something is wrong with the sweep setup, not the model.

**Acceptance criteria:**
- [ ] Run a backtest with `debug_filters=True` against `1h` data for a representative asset group (one of the `suggested_groups` from E11's live correlation run) — e.g. via `strategy/kairos_pipeline.py --stage oracle --interval 1h --group_id <id> ...` or directly via `kairos_strategies.py --interval 1h --group_id <id>` (check current CLI flags with `--help`; use whichever stage most directly exposes the orchestrator's per-day filter diagnostics).
- [ ] Collect entropy/kurtosis/volume-percentile values across enough `1h` bars (aim for at least a few hundred, matching the spirit of the `1d` calibration's sample size) to compute reasonable thresholds — same method as whatever produced the `1d` numbers in the CLAUDE.md table (look at git history / `docs/` for how those were originally derived, if documented; if not documented, use the same statistical judgment: entropy threshold near the observed distribution's typical ceiling, kurtosis_max well above typical excess kurtosis so discrete token sampling noise doesn't trip it, min_volume_percentile low enough that mean-reverting volume predictions aren't over-filtered).
- [ ] Add `_FILTER_PRESETS_BY_INTERVAL["1h"] = {"entropy_threshold": <value>, "kurtosis_max": <value>, "min_volume_percentile": <value>}` to `kairos_orchestrator.py`, with a comment recording how/when these were derived (mirror the comment style already used in `_DISABLED_BY_CLASS`'s per-strategy sharpe annotations).
- [ ] Update the CLAUDE.md "OrchestratorConfig defaults (after calibration)" table (or add a new `1h` row) documenting the new numbers and the date/method.
- [ ] Confirm `1d` behavior is completely unaffected: `_FILTER_PRESETS_BY_INTERVAL` gaining a `"1h"` key does not change `_FILTER_PRESETS_BY_INTERVAL.get("1d", {})`, which stays `{}` (empty, falling through to dataclass defaults) unless a future story explicitly adds a `"1d"` entry too.

**Definition of done:**
- [ ] Real sweep data collected and reviewed (not fabricated numbers).
- [ ] `_FILTER_PRESETS_BY_INTERVAL["1h"]` populated with justified values.
- [ ] CLAUDE.md updated.
- [ ] `uv run --with pytest python -m pytest tests/unit/ -q` still green.
- [ ] Changes committed and `docs/todo.md` E12-S02 item checked off.
