# Connecting Bitvavo

Dutch crypto exchange, registered with De Nederlandsche Bank (DNB) —
MiCA-compliant NL venue. No Kairos account exists yet; this is a
forward-looking implementation guide, not measured results (contrast
`docs/ibkr-cost-discovery.md`, which is empirical). See
`docs/broker-api-interface.md` for the `ExchangeProbe` contract this maps
to, and `docs/playbooks/add-exchange-probe.md` for the steps to actually
build `scripts/bitvavo_instruments.py`.

## Auth & account setup

- Market data (`fetchMarkets`, `fetchTicker`, `fetchOHLCV`, order book,
  trades) is public — **no API key needed**, rate-limited by IP instead of
  key.
- **Trading-fee lookup (`fetchTradingFee`/`fetchTradingFees`) is listed
  separately from the public-methods group in ccxt's docs** and is almost
  certainly account-scoped (your fee tier depends on your own trailing
  30-day volume) — treat it as needing an API key until confirmed
  otherwise. `resolve_instrument`/`get_reference_price` can be built and
  tested with zero account; `get_cost_model` cannot.
- API keys are generated from the Bitvavo web app once an account exists
  (standard KYC — this session did not go further into the KYC flow
  itself, out of scope for a scaffolding-only pass).

## Tier 1 interface mapping (Pattern B — direct fee-schedule lookup)

| `ExchangeProbe` method | Bitvavo/ccxt call |
|---|---|
| `connect(args)` | `ccxt.bitvavo({'apiKey': ..., 'secret': ...})` (omit both for public-only use) |
| `resolve_instrument(symbol)` | `exchange.load_markets()[symbol]` → `market['id']`, `market['precision']`, `market['limits']['amount']['min']` |
| `get_reference_price(instrument)` | `exchange.fetch_ticker(symbol)['last']` |
| `get_cost_model(instrument, ref_price, base_currency)` | `exchange.fetch_trading_fee(symbol)` → `{'maker': ..., 'taker': ...}`, both as fractions (multiply by 100 for `commission_rate_pct`) |
| `get_fx_rate(currency, base)` | Not needed if `base_currency='EUR'` — most Bitvavo markets quote in EUR directly |

ccxt also exposes native REST/WS methods 1:1 with Bitvavo's own API
(`docs.bitvavo.com`), including `Ws`-suffixed websocket variants for most
calls — not needed for a one-shot probe sweep.

## Fee schedule (the critical check)

**No flat floor** — this is the key finding, since IBKR's flat $1
minimum-per-order is exactly what ruled it out for Kairos's ~€18 average
trade. Bitvavo is a pure percentage schedule, 9 volume tiers:

| 30-day volume | Maker | Taker |
|---|---|---|
| < €100,000 (base tier) | 0.15% | 0.25% |
| > €100,000 | 0.10% | 0.20% |
| ... | ... | ... |
| > €25,000,000 | 0.03% | 0.04% (0% maker / 0.02% taker on major pairs, promo) |
| > ~€500,000,000 (top tier) | ~0.00% | ~0.01% |

Kairos would sit in the base tier (0.15%/0.25%) indefinitely at its current
trade volume. On an €18 trade that's ~3-5 cents round trip — no
fractional-order barrier, no per-order floor. Stablecoin-pair volume does
not count toward tier progression.

`fetchTradingFee` should return the account's *current* tier rate directly
(not something Kairos needs to compute from 30-day volume itself) — worth
confirming against a real key before trusting this.

## Order size / precision

`GET /markets`'s `minOrder`/`minOrderInQuoteAsset` fields (surfaced through
ccxt as `market['limits']['amount']['min']` /
`market['limits']['cost']['min']`) give the minimum order size per market —
per-market, not account-wide, so `resolve_instrument` must read it per
symbol, not assume one constant. Not itself measured this session.

## Rate limits

- **1000 weight points/minute**, tracked per API key if authenticated, per
  IP if not. A `createOrder` costs 1 weight point (the only per-endpoint
  weight the docs surface directly — others weren't itemized in what was
  fetched this session).
- Exceeding the limit: authenticated requests are blocked 1 minute;
  unauthenticated (public/IP-tracked) requests are blocked **15 minutes** —
  materially harsher, so an unauthenticated bulk sweep (e.g. resolving
  150+ universe symbols) should pace itself well under the ceiling rather
  than relying on the shorter authenticated cooldown as a safety net.
- WebSocket: 5000 msg/s general, 50 msg/s for "Market Data Pro" sessions.

## Gotchas / open questions

- **No sandbox/testnet found** in ccxt's Bitvavo docs. Unlike Kraken/Bybit,
  there may be no way to test order placement without live funds — a probe
  sweep (market data + fee lookup, no order placement) doesn't need one,
  but Tier 2 (execution) eventually would. Worth a direct confirmation with
  Bitvavo support before assuming this is a hard blocker.
- Fee-schedule auth requirement (above) is inferred from ccxt's doc
  grouping, not directly confirmed — verify with a real API key on first
  use.
- Per-endpoint weight costs beyond `createOrder`=1 weren't available from
  the page fetched this session; check `docs.bitvavo.com/docs/rate-limits/`
  directly before writing a high-volume sweep loop.

## Not done this session

No Bitvavo account, no API key, no real sweep run. This doc and
`scripts/exchange_probe.py`'s Pattern B scaffolding are the extent of this
pass, per Baz's 2026-09-06 decision (scaffolding + playbook now, real
sandbox/account testing later — see `docs/playbooks/add-exchange-probe.md`'s
"What this does not cover yet"). Signing up and running a real
`scripts/bitvavo_instruments.py --symbols BTC-USD,ETH-USD --report` sweep
is the natural next step whenever that's picked back up.

## Sources

- [bitvavo - CCXT](https://docs.ccxt.com/docs/exchanges/bitvavo)
- [Rate limits | Bitvavo API Docs](https://docs.bitvavo.com/docs/rate-limits/)
- [Fees | Bitvavo.com](https://bitvavo.com/en/fees)
- [Bitvavo introduces 0% maker and 0.02% taker fees for €25M+ volume fee tier](https://bitvavo.com/en/news/new-maker-taker-fees)
- [Trading Rules | Bitvavo.com](https://bitvavo.com/en/trading-rules)
- [What is the trading fee when I buy or sell crypto? – Bitvavo Help Center](https://support.bitvavo.com/hc/en-us/articles/4405175148689-What-is-the-trading-fee-when-I-buy-or-sell-crypto)
