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
  not inferred — and confirmed live 2026-09-06 (see below).
- **Confirmed environment-locked (2026-09-06 live test)**: the sandbox key
  works against `sandbox.bitstamp.net` (returns a fake balance: 0.5 BTC,
  10 ETH, 40000 USD) but the identical signed request against
  `www.bitstamp.net` (production) is flatly rejected —
  `403 {"status":"error","reason":"API key not found","code":"API0001"}`.
  Same property OKX's demo key had — a sandbox key cannot touch real funds
  regardless of what permissions it's granted.

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
for Kairos's trade size. **CONFIRMED 2026-09-06 via live
`POST /api/v2/fees/trading/` calls (sandbox key, all pairs and BTC/EUR
individually): 0.30% maker / 0.40% taker at base tier**, matching the
pre-measurement research figure exactly. Scales down to ~0.00%/0.03% past
$1B 30-day volume per the published schedule (not re-measured — sandbox
account is fixed at base tier). FX/stablecoin pairs reportedly get an 80%
fee reduction (unmeasured).

**This also settles the Advanced-vs-Instant-Buy/Sell ambiguity below**:
`/api/v2/fees/trading/` is the actual API-returned rate, not a UI quote —
the API genuinely uses the 0.30%/0.40% Advanced schedule.

**Real constraint, different in kind from IBKR's floor: minimum order size
is €10.00 EUR for BTC/EUR and XRP/EUR**, confirmed live 2026-09-06 via the
public `GET /api/v2/trading-pairs-info/` endpoint (no auth needed).
Kairos's ~€18 average trade clears it, but not by much; a smaller signal
could be blocked outright rather than just taxed. Worth checking against
the actual signal-size distribution before assuming every Kairos order
would clear this floor.

~~One unconfirmed wrinkle: Bitstamp's consumer UI has historically offered
a separate "Instant Buy/Sell" flow with a spread-based fee... flag for
confirmation once a real key exists.~~ Resolved above — the API uses the
Advanced schedule.

## Order size / precision

Confirmed live 2026-09-06 via `GET /api/v2/trading-pairs-info/` (public,
no auth): returns `minimum_order`, `base_decimals`, `counter_decimals`
directly per pair (e.g. BTC/EUR: `minimum_order: "10.00 EUR"`,
`base_decimals: 8`, `counter_decimals: 2`) — maps cleanly to
`InstrumentMeta.min_size`/`min_tick`. Only BTC/EUR and XRP/EUR pulled so
far, not a full sweep. Note some pairs report `trading: "Disabled"` in
this same response (e.g. ETH/BTC in the sandbox) — `resolve_instrument`
should check this field, not assume every listed pair is tradeable.

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

## What's been probed for real, and what's still open

**Done 2026-09-06**: ID verification cleared, Baz generated a sandbox API
key same day. 5 live calls succeeded against `sandbox.bitstamp.net`:
`GET /api/v2/ticker/btceur/` (get_reference_price, Ok),
`POST /api/v2/account_balances/` (auth check, Ok),
`POST /api/v2/fees/trading/` and `.../btceur/` (get_cost_model, Ok — the
confirmed 0.30%/0.40% figure above), and
`GET /api/v2/trading-pairs-info/` (resolve_instrument sizing, Ok — the
confirmed €10 minimum). Also confirmed the sandbox key is
environment-locked against production (see Auth above). This settled the
fee and minimum-order questions without building a full probe script,
same as OKX's approach.

**Still open**: no `scripts/bitstamp_instruments.py` exists yet (would
formalize this into the shared `ExchangeProbe` framework per
`docs/playbooks/add-exchange-probe.md`); only BTC/EUR and XRP/EUR checked
for order size, not a full sweep; rate-limit figures still disagree
(above) and weren't tested live; fee scaling at higher volume tiers is
unmeasured (sandbox account is fixed at base tier); no production
(non-sandbox) Kairos-owned account or key exists yet — this was a
sandbox-only smoke test.

## Sources

- **Live sandbox API calls, 2026-09-06** (Baz's own Bitstamp sandbox
  account, `sandbox.bitstamp.net`) — `/api/v2/ticker/btceur/`,
  `/api/v2/account_balances/`, `/api/v2/fees/trading/` (all pairs and
  BTC/EUR), `/api/v2/trading-pairs-info/`, plus the same signed call
  repeated against `www.bitstamp.net` to confirm environment-locking.
  Primary source for the confirmed 0.30%/0.40% fees, the confirmed €10
  minimum order, and the Advanced-vs-Instant-Buy/Sell resolution.
  Credentials used only as environment variables to a throwaway
  scratchpad script, never written to any repo file.
- [Bitstamp API](https://www.bitstamp.net/api/)
- [Fee schedule – Bitstamp by Robinhood](https://www.bitstamp.net/fee-schedule/)
- [Bitstamp 2026; Fees [From 0.30%] & User Levels — TradingFinder](https://tradingfinder.com/exchanges/bitstamp/)
- [bitstamp - ccxt](https://docs.ccxt.com/docs/exchanges/bitstamp)
- [Robinhood completes $200 million acquisition of crypto exchange Bitstamp | The Block](https://www.theblock.co/post/356701/robinhood-bitstamp-acquisition)
- [Robinhood Brings TradFi Perps to Europe — Bankless Times](https://www.banklesstimes.com/articles/2026/07/02/robinhood-brings-tradfi-perps-to-europe-accelerates-blockchain-and-ai-trading-push/)
