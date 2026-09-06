# Phase 5 — Automated execution

**Goal:** act on recommendations automatically via exchange/broker API.
**Gated on:** Phase 4 results (paper trading matches backtest).
**Rough effort:** 4-6 subagent-days (Sonnet 5), staged rollout over weeks.

## Tasks

### 5.1 Broker abstraction + crypto first
- `kairos/broker.py` interface: `place_order`, `close`, `get_positions`,
  `get_balance` — this is Tier 2 in
  [`docs/broker-api-interface.md`](../docs/broker-api-interface.md),
  undesigned as of 2026-09-06. Tier 1 (cost/tradeability discovery, the
  prerequisite question of whether a broker is even worth wiring up) is
  built — see that doc and [`docs/exchanges/`](../docs/exchanges/).
- First implementation: crypto exchange via `ccxt`. **Binance is out** —
  it failed to secure an EU MiCA license and shut off EU users
  2026-07-01, so it cannot serve a NL-resident account regardless of API
  quality. Candidates measured 2026-09-06:
  Bitvavo / Kraken / Bybit EU (all MiCA-compliant, ccxt-supported,
  percentage-fee — not IBKR's flat-floor problem) and Finst (Dutch,
  0.15% flat, ccxt support unconfirmed). See `docs/exchanges/*.md`.
- Equities later via Alpaca or IBKR. **Alpaca launched a real European
  broker in April 2026** (Alpaca Europe, MiFID II-compliant; Xetra live,
  Euronext/LSE "expected to follow" — not confirmed for Euronext
  Amsterdam yet) — but it turned out to be **Broker-as-a-Service, not a
  direct trading account**: the EU product onboards Kairos as its own
  broker-of-record (KYC-as-a-service, correspondent-set fees), not as a
  self-directed account like IBKR. Materially heavier integration than
  assumed when this was first flagged — see
  `docs/exchanges/alpaca-europe.md` before committing to it. IBKR was
  measured and ruled out for Kairos's trade size — see
  `docs/ibkr-cost-discovery.md`. Avoid the automated-browser route unless
  a broker truly has no API; it is by far the most fragile option.
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
