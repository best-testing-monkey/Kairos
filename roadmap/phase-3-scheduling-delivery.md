# Phase 3 — Scheduling & delivery

**Goal:** reports arrive unattended — daily and hourly — via Telegram first,
then a web page.

**Depends on:** Phase 2.
**Rough effort:** 3-4 subagent-days (Sonnet 5 + Haiku for styling).

## Tasks

### 3.1 Scheduler
- Cron entries invoking `kairos_live.py --profile ... --notify`:
  daily after each market close / 00:00 UTC for crypto; hourly on 1h profiles
  at bar close + a small delay for data availability.
- Retry-with-backoff on yfinance failures. A failed run must **alert**, never
  silently skip.
- Recommendation: cron + CLI entrypoint (simple, restart-safe) over a custom
  daemon.
- Owner: 1 Sonnet subagent.

### 3.2 Telegram bot (first delivery channel — before the web page)
- `python-telegram-bot`; push each report to a configured chat_id.
- Commands: `/report` (latest), `/positions`, `/pause <profile>`, `/status`.
- Secrets via env / `.env` (never committed; add `.env` to `.gitignore`).
- Tests: message formatting, command routing with the API mocked.
- Owner: 1 Sonnet subagent.

### 3.3 Web page
- Small FastAPI app: latest report per profile, open positions, equity curve of
  followed advice, historical reports. Server-rendered HTML, no SPA. Optional
  basic auth.
- Owner: 1 Sonnet subagent (+ Haiku for styling).

### 3.4 Hourly-interval validation (prerequisite for hourly reports)
- Rerun pipeline stages 3-4 at `1h` for the top groups (yfinance caps 1h
  history at 729 days — sufficient). Hourly reports go live only for profiles
  with a positive validated 1h profile.
- Owner: orchestrator (analysis work, run directly).

## Exit criteria
Daily + hourly Telegram messages arrive unattended for a full week with no
missed runs.
