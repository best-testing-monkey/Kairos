# Phase 5 — Automated execution

**Goal:** act on recommendations automatically via exchange/broker API.
**Gated on:** Phase 4 results (paper trading matches backtest).
**Rough effort:** 4-6 subagent-days (Sonnet 5), staged rollout over weeks.

## Tasks

### 5.1 Broker abstraction + crypto first
- `kairos/broker.py` interface: `place_order`, `close`, `get_positions`,
  `get_balance`.
- First implementation: crypto exchange via `ccxt` (Binance / Kraken / Bybit) —
  cleanest APIs, 24/7 markets, and matches the best-performing asset class.
- Equities later via Alpaca or IBKR. Avoid the automated-browser route unless a
  broker truly has no API; it is by far the most fragile option.
- Owner: 1 Sonnet subagent per broker adapter.

### 5.2 Risk guardrails (non-negotiable, before the first live order)
- Max position size per asset; max total exposure; max daily loss kill-switch.
- Per-order sanity checks (price within N% of last close), duplicate-order
  protection, `--dry-run` flag defaulting **ON**.
- Every order and every rejection logged and mirrored to Telegram.
- Owner: 1 Sonnet subagent; reviewed line-by-line before enabling.

### 5.3 Staged rollout
1. Exchange testnet/sandbox.
2. Tiny real capital on the single best crypto profile.
3. Scale profile-by-profile only after each stage matches paper-trade
   expectations.
- Telegram `/halt` command stops all trading immediately.

## Data source note
yfinance is unofficial and rate-limited — acceptable for daily reports,
marginal for hourly, wrong for execution. In this phase, price data should come
from the exchange itself (ccxt provides OHLCV for free).
