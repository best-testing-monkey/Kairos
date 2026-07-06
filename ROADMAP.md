# Kairos Development Roadmap

Goal: Kairos autonomously produces **daily and hourly trading reports**
(open/close advice with entry, stop, and target prices), delivered via
**web page and/or Telegram**, and eventually **acts on them automatically**
via exchange API.

Implementation is executed by **Sonnet 5 subagents** (Haiku for simple or
mechanical tasks); the orchestrating session handles analysis, review, and git.

## Where we are (2026-07)

Done:
- Backtest engine, 30+ strategies, meta-filters, per-profile/per-class
  disabled-strategy lists.
- 5-stage discovery pipeline (`strategy/kairos_pipeline.py`, docs in
  `strategy/PIPELINE.md`): universe screen (124/153 pass) -> correlation
  (28 groups) -> oracle (27 groups) -> base model (GPU) -> finetuned.
  Results persisted to SQLite (`data/pipeline_results.db`) + CSV.
- GPU backtest at ~0.4 s/iteration; 208 unit tests (no GPU/network needed).
- **In progress: Stage 4 (base model) runs** — 101 base runs, ~3,000
  `model_results` rows so far.

Everything below assumes Stage 4/5 conclude with a vetted set of
`(asset group, interval, enabled strategies)` profiles.

## Phases

| Phase | File | Depends on | Effort (subagent-days) |
|---|---|---|---|
| 1. Selection & config layer | [roadmap/phase-1-selection-config.md](roadmap/phase-1-selection-config.md) | Stage 4 done | 2-3 |
| 2. Live signal generation | [roadmap/phase-2-live-signals.md](roadmap/phase-2-live-signals.md) | Phase 1 | 3-4 |
| 3. Scheduling & delivery (Telegram, web) | [roadmap/phase-3-scheduling-delivery.md](roadmap/phase-3-scheduling-delivery.md) | Phase 2 | 3-4 |
| 4. Paper trading & accounting | [roadmap/phase-4-paper-trading.md](roadmap/phase-4-paper-trading.md) | Phase 3 | 2-3 (+4-6 wks data) |
| 5. Automated execution | [roadmap/phase-5-auto-execution.md](roadmap/phase-5-auto-execution.md) | Phase 4 results | 4-6 |

The first user-visible win is the end of Phase 3: a daily Telegram report.
That is ~8-11 subagent-days of implementation from today, and none of it is
blocked on Stage 5 finetuning.

## Cross-cutting concerns (ongoing, mostly Haiku-sized)

- **Ops hygiene**: structured logging, run-log table in SQLite, rotation of
  `results/` CSVs (hundreds already accumulating).
- **Tests**: every phase lands with unit tests; preserve the no-GPU/no-network
  property of `tests/unit/`.
- **Data source risk**: yfinance is unofficial and rate-limited — fine for
  daily reports, marginal for hourly, wrong for execution. Phase 5 switches
  price data to the exchange itself (ccxt).
- **Secrets**: all tokens/keys via environment; `.env` in `.gitignore` from
  the start.
