# Connecting Bitstamp (Tier 1 — cost/tradeability discovery)

Luxembourg-licensed crypto exchange (Bitstamp Europe S.A., CSSF-authorized
under MiCA since May 2025), founded 2011 — one of the oldest exchanges
still operating. Acquired by Robinhood for $200M (closed June 2025);
Robinhood has kept Bitstamp's own operations running (used for smart order
routing) and is actively expanding EU product on top of it (perpetual
futures launched July 2026 via the separate Robinhood Europe UAB /
Lithuania entity) — no sign of Bitstamp's core spot business being wound
down. No Kairos account exists yet; this is how to connect it, not a
measured result (see `docs/broker-api-interface.md` for the Tier 1/Tier 2
split, `docs/ibkr-cost-discovery.md` for the empirical-measurement
precedent this doc sets up but doesn't duplicate).

## Auth & account setup

- Public market data (`fetchMarkets`, `fetchTicker`, `fetchOrderBook`,
  `fetchTrades`) needs no key — self-service, same as Bitvavo/Kraken/Bybit
  EU, not gated like Finst.
- Private endpoints (trading, `fetchTradingFee`/`fetchTradingFees`) need an
  API key/secret from account settings, signed per-request: `X-Auth`,
  `X-Auth-Signature` (HMAC-SHA256), `X-Auth-Nonce`, `X-Auth-Timestamp`
  (must be within 150s of server time), `X-Auth-Version: v2`. Standard
  self-service signup, no sales conversation.
- **Real sandbox exists**: `https://sandbox.bitstamp.net/api/v2/` mirrors
  the full production API surface. Confirmed from Bitstamp's own API docs,
  not inferred.

## Tier 1 interface mapping (Pattern B — direct fee-schedule lookup)

| `ExchangeProbe` method | Bitstamp/ccxt call | Auth needed |
|---|---|---|
| `resolve_instrument(symbol)` | ccxt `load_markets()` / `market(symbol)` | No |
| `get_reference_price(instrument)` | ccxt `fetch_ticker(symbol)` | No |
| `get_cost_model(instrument, ref_price, base)` | ccxt `fetch_trading_fee(symbol)` (Bitstamp is a first-class ccxt exchange, this is a native unified method) | Yes |
| `get_fx_rate(currency, base)` | Not generally needed — Bitstamp lists EUR pairs directly (e.g. `BTC/EUR`); only relevant off a non-EUR quote | — |

Confirmed Pattern B — one `fetch_trading_fee` call, no dry-run/whatIf probing.

## Fee schedule (the critical check)

**No flat commission floor per order** — same property that ruled IBKR out
for Kairos's trade size. Base tier (< $10,000 30-day USD-equivalent
volume): **0.30% maker / 0.40% taker**, scaling down to ~0.00%/0.03% past
$1B volume. FX/stablecoin pairs get an 80% fee reduction. Sources converge
on this figure (tradingfinder.com and a second independent search both
landed on 0.30%/0.40%), unlike Kraken's conflicting numbers — reasonably
confident here, but still pull it live before trusting it for real, same
discipline as every other doc in this directory.

**Real constraint, different in kind from IBKR's floor: minimum order size
is €10 / $10 / £10 (or 0.0002 BTC / 0.002 ETH directly)**, confirmed from
Bitstamp's own API docs. Not a commission problem — a notional-size gate.
Kairos's ~€18 average trade clears it, but not by much; a smaller signal
could be blocked outright rather than just taxed. Worth checking against
the actual signal-size distribution before assuming every Kairos order
would clear this floor.

One unconfirmed wrinkle: Bitstamp's consumer UI has historically offered a
separate "Instant Buy/Sell" flow with a spread-based fee (materially
higher than the Advanced/API trading fee schedule above) — not found
re-confirmed for 2026 in this pass. The API almost certainly routes
through the Advanced trading engine (the 0.30%/0.40% schedule), not the
consumer instant-buy flow, but this wasn't independently verified against
API-specific fee docs — flag for confirmation once a real key exists.

## Order size / precision

Minimum order size above (€10/$10/£10 or crypto-equivalent) is the one
hard figure found. Per-pair tick/lot precision not measured — pull via
ccxt `market['precision']`/`market['limits']` same as every other
exchange in this directory.

## Rate limits

**Sources disagree, flagging rather than picking one**: Bitstamp's own API
docs state 400 requests/second with a 10,000-requests-per-10-minutes
default threshold (higher available via bespoke agreement). A secondary
source (encountered while checking ccxt's Bitstamp integration) instead
cited 8,000 requests/10 minutes. Trust the official page's 400/s +
10,000/10min figure until proven otherwise — it's the primary source; the
8,000 figure may simply be stale.

## Gotchas

- Rate-limit figures disagree across sources (above) — verify against
  `bitstamp.net/api/` directly before writing a high-frequency sweep loop.
- The €10 minimum order size is close enough to Kairos's average trade
  that it's worth checking against real signal sizes, not just noting in
  passing.
- Two possible fee schedules (Advanced/API vs. consumer Instant Buy/Sell)
  weren't fully disambiguated this pass — confirm the API genuinely uses
  the 0.30%/0.40% schedule, not something spread-based, on first real use.

## What's needed before this can be probed for real

No Kairos-owned Bitstamp account or API key exists — scaffolding + this
connection doc only, per Baz's 2026-09-06 scope decision (see
`docs/playbooks/add-exchange-probe.md`). Unlike Kraken (no self-service
spot sandbox) and similar to Bybit EU, **Bitstamp's sandbox needs no
funded account at all** — this is genuinely cheap to smoke-test whenever
that's picked back up: register, generate a sandbox key, implement
`scripts/bitstamp_instruments.py` against the mapping above, run it
against `sandbox.bitstamp.net` first.

## Sources

- [Bitstamp API](https://www.bitstamp.net/api/)
- [Fee schedule – Bitstamp by Robinhood](https://www.bitstamp.net/fee-schedule/)
- [Bitstamp 2026; Fees [From 0.30%] & User Levels — TradingFinder](https://tradingfinder.com/exchanges/bitstamp/)
- [bitstamp - ccxt](https://docs.ccxt.com/docs/exchanges/bitstamp)
- [Robinhood completes $200 million acquisition of crypto exchange Bitstamp | The Block](https://www.theblock.co/post/356701/robinhood-bitstamp-acquisition)
- [Robinhood Brings TradFi Perps to Europe — Bankless Times](https://www.banklesstimes.com/articles/2026/07/02/robinhood-brings-tradfi-perps-to-europe-accelerates-blockchain-and-ai-trading-push/)
