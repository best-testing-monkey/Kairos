# Connecting OKX (Tier 1 — cost/tradeability discovery)

Global crypto exchange. OKX Europe Limited holds a MiCA CASP license from
Malta's MFSA (authorised January 2025) plus a Malta payments-institution
license (February 2026), passported across all 30 EEA states including the
Netherlands. No Kairos account exists yet — this is how to connect it, not
a measured result (see `docs/broker-api-interface.md` for the Tier 1/Tier 2
split, `docs/ibkr-cost-discovery.md` for the empirical-measurement
precedent this doc is meant to set up, not duplicate).

**Spot only is the relevant scope here.** EU regulators (NL included) have
progressively restricted retail access to crypto derivatives since 2023;
OKX's answer is X-Perps, a separately MiCA-licensed perpetual product for
EU retail — not needed for Kairos's Tier 1 discovery (no margin/derivatives
involved), but worth knowing if this ever goes beyond spot.

## Auth & account setup

- Public market-data endpoints (instruments, tickers, candles, order book)
  need no key — IP-rate-limited instead.
- Trading-fee lookup is account-scoped (VIP tier depends on your own
  volume/balance) — same pattern as every other Pattern B exchange here,
  treat `get_cost_model` as needing a key until confirmed otherwise.
- **Real demo/sandbox trading API, separate from Kraken's spot gap and
  Bitvavo's unknown**: OKX → Trade → Demo Trading → Personal Center →
  create a **separate demo API key**. Demo trading mirrors live price data
  and the full production API shape (not stale/limited like some
  competitors' testnets) — genuinely usable for a real Tier 1 smoke test
  with zero funded account. One header requirement: every demo request
  needs `x-simulated-trading: 1` alongside the normal
  `OK-ACCESS-KEY`/`SIGN`/`PASSPHRASE`/`TIMESTAMP` headers.
- Live account (beyond demo) needs standard KYC.

## Tier 1 interface mapping (Pattern B — direct fee-schedule lookup)

| `ExchangeProbe` method | OKX/ccxt call | Auth needed |
|---|---|---|
| `resolve_instrument(symbol)` | ccxt `load_markets()` / `market(symbol)` (native: `GET /api/v5/public/instruments`) | No |
| `get_reference_price(instrument)` | ccxt `fetch_ticker(symbol)` (native: `GET /api/v5/market/ticker`) | No |
| `get_cost_model(instrument, ref_price, base)` | ccxt `fetch_trading_fee(symbol)` — OKX's implementation supports spot/swap/future/option (native: `GET /api/v5/account/trade-fee`) | Yes, for your real tier |
| `get_fx_rate(currency, base)` | Not generally needed — OKX lists EUR-quoted pairs directly; only relevant if quoting outside account base currency | — |

Confirmed Pattern B: one `fetch_trading_fee` call is the whole cost lookup,
same mechanism as Bitvavo/Kraken/Bybit EU. No dry-run order needed.

## Fee schedule (the critical check)

**No flat minimum commission per order** — pure percentage, VIP-tiered.
Base (Regular) spot tier confirmed as of mid-2026: **0.080% maker / 0.100%
taker** — matches the figure from earlier broader research, and among the
cheapest base-tier rates found across all candidates so far (only Bybit EU's
0.10%/0.10% is close; Bitvavo/Kraken are higher at base tier). Fee framework
was restructured April 2026 (8→9 VIP levels, lower entry thresholds); rates
drop further with volume (VIP3 taker ~0.028%) and top tiers see negative
(rebate) maker fees. On Kairos's ~€18 average trade, base-tier taker cost is
under 2 cents — not remotely close to IBKR's flat-$1 problem.

## Order size / precision

Not found in this pass — no minimum-order-size figure surfaced in the
research done here. Pull per-pair via `load_markets()`'s `precision`/
`limits` fields (same `InstrumentMeta.min_tick`/`size_increment`/`min_size`
mapping as every other Pattern B exchange) once actually probing.

## Rate limits

- **Global cap: 250 requests/second (500 per 2s)** across all REST calls —
  generous for a probe sweep.
- Public market-data endpoints: ~20 requests/2s (tickers, order book,
  candles, instrument listings) — IP-scoped, shared across anything running
  on the same host/IP.
- Private endpoints (fee lookup, orders) are User-ID-scoped, independent
  per account: single-order 60/2s, batch 300/2s (not relevant to Tier 1,
  which never places an order).
- Known ccxt friction: recreating the exchange instance on every call
  resets ccxt's internal rate limiter and can trip OKX's actual limit
  faster than expected (error 50011, "Requests too frequent") — reuse one
  `ccxt.okx()` instance for the whole sweep, don't reinstantiate per symbol.

## Gotchas

- Demo trading needs its own **separate** API key from any live key, and
  every request needs the `x-simulated-trading: 1` header — a live key
  used against demo endpoints (or vice versa) will reject outright rather
  than silently mixing up which environment it hit.
- `fetchTradingFee` in ccxt's OKX implementation only covers spot/swap/
  future/option markets — fine for Kairos's spot-only scope, just don't
  assume it covers every OKX product type.
- Public vs. authenticated fee-tier distinction is the same trap as
  Kraken's doc describes: an unauthenticated probe would need to fall back
  to a published base-tier number rather than the account's real rate.

## What's needed before this can be probed for real

Signup in progress (Baz, 2026-09-06). Real first-hand data point so far:
OKX reported **"5 minutes to prepare your account for identity
verification"** — sounds meaningfully faster to at least get started on
than Bybit EU, which gates on up to 3 days of ID verification plus a
separate 48h API-key wait (see `docs/exchanges/bybit-eu.md` and
`docs/exchanges/README.md`'s "Real signup latency" note). Full end-to-end
verification time for OKX not yet confirmed — only the initial prep-step
duration is known so far. Next step per
`docs/playbooks/add-exchange-probe.md` once verification clears: register
demo trading, generate a demo API key, implement `scripts/okx_instruments.py`
against the mapping above, run it.

## Sources

- [OKX Fees Guide 2026: Spot, Futures, Withdrawals](https://tradersunion.com/brokers/crypto/view/okex/fees/)
- [OKX fees: 0.020% maker, 0.050% taker (2026)](https://feeedge.com/exchanges/okx)
- [OKX API guide | OKX technical support](https://www.okx.com/docs-v5/en/)
- [OKX V5 Demo Trading API Guide for Traders](https://voiceofchain.com/academy/okx-v5-demo-trading-api)
- [OKX Demo Trading with Virtual Funds Guide](https://blockchain8.hashnode.dev/okx-demo-trading-virtual-funds-guide-2025)
- [okx - CCXT Docs](https://docs.ccxt.com/docs/exchanges/okx)
- [OKX X-Perps: How a 5-Year Expiry Clause Cracked Europe's Derivatives Market](https://blockeden.xyz/blog/2026/04/17/okx-x-perps-europe-mifid-5-year-expiry-perpetual-futures/)
- [OKX Restricted Countries List 2026](https://www.coinperps.com/learn/okx-restricted-countries)
- [Does OKX have a MiCA (CASP) license? Yes, licensed in Malta](https://casptracker.eu/exchange/okx/)
