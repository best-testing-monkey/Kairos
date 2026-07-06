# Phase 2 — Live signal generation ("today mode")

**Goal:** the backtest answers "how would this have done"; production needs
"what do I do *now*". Produce structured, actionable recommendations from the
latest data.

**Depends on:** Phase 1.
**Rough effort:** 3-4 subagent-days (Sonnet 5).

## Tasks

### 2.1 Live inference mode
- `run_live(profile) -> list[Recommendation]` in `strategy/kairos_live.py`:
  fetch latest bars, run **one** prediction step (no walk-forward), apply
  enabled strategies + meta-filters, emit:
  ```
  Recommendation:
    symbol, action (open_long | open_short | close | hold),
    size_fraction (Kelly-derived), entry, stop, target,
    confidence, strategy, distribution_stats, timestamp
  ```
- Data freshness guard: refuse to emit if the latest bar is stale, aware of
  24/7 vs market-hours assets (reuse `is_24_7_crypto_symbol`).
- Tests: recommendation schema, staleness guard, no-signal days produce an
  explicit "no action" result (never an empty silent exit).
- Owner: 1 Sonnet subagent.

### 2.2 Position state store
- SQLite `data/positions.db`: open positions, recommendation history, outcome
  once closed. Live mode must know what's open to be able to say "close X" —
  this is the persistent counterpart to the backtester's in-memory portfolio.
- Reconciliation of recommendation -> position lifecycle
  (recommended -> opened -> closed -> outcome recorded).
- Tests: lifecycle transitions on a temp DB.
- Owner: 1 Sonnet subagent.

### 2.3 Report generator
- `strategy/kairos_report.py`: render recommendations + open-position status +
  short performance recap into:
  - (a) Markdown/plain text (Telegram-ready),
  - (b) a self-contained HTML page.
- Include a "why" per recommendation: strategy name, EV, entropy/kurtosis
  readings from the distribution.
- Owner: 1 Sonnet subagent (Haiku can do the HTML templating pass).

## Exit criteria
`uv run strategy/kairos_live.py --profile X` prints a dated report with
concrete actionable positions and records them in `data/positions.db`.
