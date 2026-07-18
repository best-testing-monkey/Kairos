# Kairos operating playbooks

This is the index for running Kairos day to day: finding effective
strategies, generating actionable signals, and deciding what to do with
them. Each playbook is a self-contained runbook — commands, what to expect,
and troubleshooting. Deep reference material (flags, DB schema, formulas)
lives in [`strategy/PIPELINE.md`](../../strategy/PIPELINE.md),
[`strategy/README.md`](../../strategy/README.md), and
[`docs/rfc_allocation_sheet.md`](../rfc_allocation_sheet.md) — the
playbooks link out to those instead of duplicating them.

## System overview

Kairos discovers which assets and strategies are worth trading, then turns
that into a report you act on manually. The pipeline (`strategy/kairos_pipeline.py`)
runs in stages: **universe** (liquidity/volatility screen) → **correlation**
(group co-moving symbols) → **oracle** (perfect-foresight ceiling per
strategy, no GPU — actual next-bar OHLCV instead of a model prediction) →
**base** (the same backtest against the real Kronos model, GPU required) →
a **viability report** persisted to SQLite (`data/pipeline_results.db`) with
point-in-time CSV mirrors in `results/`. The **oracle** stage also
auto-maintains a `disabled_strategies` table in the same DB, so
underperforming strategies are silenced (and later automatically
re-enabled) without any hand-editing. `strategy/kairos_signals.py` reads
the latest viability report and turns viable strategies into a **signals
report** (entries/stops/targets, expected value, an Allocation sheet). A
human reviews the report, edits the Allocation sheet's Enabled column as a
veto, and places orders by hand — nothing downstream is automated yet.

## Cadence

| Routine | Playbook | Command summary | Typical trigger | GPU needed? |
|---|---|---|---|---|
| Weekly strategy discovery | [weekly-strategy-discovery.md](weekly-strategy-discovery.md) | `kairos_pipeline.py --stage auto` (1d/6m, then 1h/3m) | Once a week | Yes (base stage) |
| Daily signals | [daily-signals.md](daily-signals.md) | `kairos_signals.py --intervals 1d --xlsx` | Daily, after 00:00 UTC (crypto) / after close (equities-FX) | Yes |
| Hourly signals | [hourly-signals.md](hourly-signals.md) | `kairos_signals.py --intervals 1h --xlsx` | Hourly, a few minutes past the top of the hour | Yes |
| Signal handling | [signal-handling.md](signal-handling.md) | n/a — reviewing report + Allocation sheet | After every signals run | No |

## Prerequisites (all playbooks)

- `uv` installed; run everything as `uv run ...` from the repo root.
- GPU available for base/finetuned pipeline stages and for `kairos_signals.py`
  (both do real Kronos inference). See "GPU recovery" in
  [`CLAUDE.md`](../../CLAUDE.md) if CUDA isn't visible.
- `data/pipeline_results.db` exists and has at least one completed weekly
  run before the daily/hourly signals playbooks will produce anything.

## Known documentation gaps

- [`strategy/DESIGN_DOC_pipeline_automation.md`](../../strategy/DESIGN_DOC_pipeline_automation.md)
  is stale: it describes the pipeline as unimplemented, but `--stage auto`
  (universe → correlation → oracle → base, chained, with a viability report)
  already exists and is what these playbooks use. Treat `strategy/PIPELINE.md`
  as the current reference instead.
- Nothing here is scheduled today: no cron job, systemd timer, or `kairos_live.py`
  exists yet. Every playbook below ends with an "Automation opportunities"
  section describing what a future scheduler could take over — see
  [`ROADMAP.md`](../../ROADMAP.md) Phases 2 (live inference) and 3
  (scheduling & Telegram delivery), which are not yet built.
