# Phase 1 — Selection & configuration layer

**Goal:** turn pipeline output into a durable, machine-readable "what to trade"
config that everything downstream reads.

**Depends on:** Stage 4 (base model) pipeline runs complete.
**Rough effort:** 2-3 subagent-days (Sonnet 5).

## Tasks

### 1.1 Stage-4 analysis & profile selection
- New script `strategy/kairos_select.py`: read `model_results` from
  `data/pipeline_results.db`, rank groups by realized Sharpe / win-rate /
  trade-count **with the base model** (not oracle numbers), apply the same
  per-class disable rules used for `_DISABLED_BY_CLASS`, emit a ranked shortlist.
- Output: `config/profiles.yaml` — list of production profiles:
  ```yaml
  - name: crypto-majors-1d
    assets: [BTC-USD, ETH-USD, SOL-USD]
    interval: 1d
    enabled_strategies: [...]        # or disabled_strategies
    position_sizing: kelly_fraction  # + params
    last_validated_run_id: <run_id>
  ```
- Tests: selection rules against a fixture SQLite DB (no GPU/network).
- Owner: 1 Sonnet subagent.

### 1.2 Config-driven runner
- Extract the `__main__` demo logic of `strategy/kairos_strategies.py` into a
  callable `run_profile(profile) -> results dict` in new
  `strategy/kairos_runner.py`. No behavior change.
- Profiles from `config/profiles.yaml` replace the hardcoded
  `_DISABLED_BY_PROFILE` lookup (keep the dicts as fallback when no yaml entry).
- Tests: `run_profile` output equivalence vs current CLI on the default profile.
- Owner: 1 Sonnet subagent.

### 1.3 Finetuned-model stage decision
- Run pipeline Stage 5 on the top ~5 groups. Keep finetuning in the production
  path only if it beats base on held-out Sharpe; otherwise mark the stage
  optional in `strategy/PIPELINE.md`.
- Owner: orchestrator (analysis), Sonnet subagent if code changes needed.

## Exit criteria
`uv run strategy/kairos_runner.py --profile crypto-majors-1d` reproduces
today's demo results, driven entirely by config.
