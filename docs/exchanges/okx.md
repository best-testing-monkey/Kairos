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
  `OK-ACCESS-KEY`/`SIGN`/`PASSPHRASE`/`TIMESTAMP` headers. **Confirmed
  environment-locked** (2026-09-06 live test): the same key + demo header
  returns a fake balance (1 BTC, 5000 USDT, ~98.8k total simulated equity);
  the identical request *without* the header is flatly rejected —
  `401 {"code":"50101","msg":"APIKey does not match current environment."}`.
  A demo key cannot touch real funds no matter what permissions it's
  granted (Trade/Withdraw included) — it simply isn't accepted outside the
  simulated environment.
- Live account (beyond demo) needs standard KYC.
- **EEA-critical gotcha, confirmed 2026-09-06, cost real debugging time**:
  an EEA-registered account (Netherlands included) only works against
  **`eea.okx.com`**, not `www.okx.com`. Hitting the wrong host doesn't
  produce an obviously-domain-related error — it returns `401
  {"code":"50119","msg":"API key doesn't exist"}` on *every* authenticated
  call, identically regardless of the demo header, which reads exactly
  like a bad/expired key. Public (no-auth) endpoints work fine on either
  host, which makes the failure mode more confusing, not less — the
  break only shows up once you try an authenticated call. If `50119` shows
  up and the key was just created and definitely pasted correctly, check
  the host before anything else.

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
**CONFIRMED 2026-09-06 via a live `GET /api/v5/account/trade-fee` call
(demo account, Lv1/base tier, `eea.okx.com`): 0.20% maker / 0.35% taker
for BTC-EUR spot.** This is meaningfully **higher** than the 0.08%/0.10%
figure the earlier research pass found — that figure was for global OKX;
**OKX Europe (the MiCA-licensed EEA entity) runs its own, higher fee
schedule**, the same pattern already seen with Bybit EU's separate legal
entity potentially differing from global Bybit. Trust the measured EEA
number over the pre-measurement research figure below it in this doc.

Still **no flat floor** — on Kairos's ~€18 average trade, 0.35% taker is
~6 cents, not remotely close to IBKR's or Saxo's problem, just not as
cheap as first assumed. Fee framework was restructured April 2026 (8→9 VIP
levels globally); whether OKX EEA's tiers scale the same way with volume
is unconfirmed — only the Lv1/base rate was measured.

**Sign-convention gotcha, worth getting right in `get_cost_model`**: the
API returns fees as **negative for a commission you pay, positive for a
rebate** — the opposite of the naive reading. `"maker": "-0.002"` means
you pay 0.2%, not that OKX pays you. Take the absolute value when mapping
into `CostModel.commission_rate_pct`, and watch for a sign-confused
mis-mapping if a maker fee ever comes back positive (a real rebate, not a
bug) at a higher VIP tier.

*(Pre-measurement research, kept for context, less reliable than the
number above): "Regular"/global spot tier was reported at 0.080% maker /
0.100% taker; rates reportedly drop further with volume (VIP3 taker
~0.028%) with negative (rebate) maker fees at top tiers — unconfirmed
whether this applies to the EEA entity's own schedule.*

## Order size / precision

Confirmed shape 2026-09-06 (`GET /api/v5/public/instruments`, BTC-EUR
only — not a sweep): the response carries `tickSz` (price tick),
`lotSz` (size increment), and `minSz` (minimum order size) directly,
mapping cleanly to `InstrumentMeta.min_tick`/`size_increment`/`min_size`.
BTC-EUR's own values weren't recorded at the time; re-pull per-pair when
building the real probe script rather than trusting a remembered number.

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

## What's been probed for real, and what's still open

**Done 2026-09-06**: full signup-to-live-call cycle completed same day —
faster end-to-end than Bybit EU's multi-day path. Baz created a demo API
key ("kairos" label, Trade/Withdraw/Read permissions — broader than
needed, but confirmed harmless on a demo-locked key, see Auth above), and
4 live calls succeeded once the EEA-host gotcha above was found and
fixed: `GET /api/v5/public/instruments` (resolve_instrument, Ok),
`GET /api/v5/market/ticker` (get_reference_price, Ok),
`GET /api/v5/account/config` (auth check, Ok), and
`GET /api/v5/account/trade-fee` (get_cost_model, Ok — the confirmed
0.20%/0.35% figure above). This settled the fee question without
building a full probe script.

**Still open**: no `scripts/okx_instruments.py` exists yet (would
formalize this into the shared `ExchangeProbe` framework per
`docs/playbooks/add-exchange-probe.md`); only one instrument (BTC-EUR)
checked, not a sweep; order size/precision was returned by the
instruments call (`tickSz`/`lotSz`/`minSz` fields exist) but not yet
extracted into this doc; whether OKX EEA's fee tiers drop with volume the
same way global OKX's do is unconfirmed.

## Sources

- **Live SIM/demo API calls, 2026-09-06** (Baz's own OKX demo account,
  `eea.okx.com`) — `/api/v5/public/instruments`, `/api/v5/market/ticker`,
  `/api/v5/account/config`, `/api/v5/account/trade-fee`,
  `/api/v5/account/balance` (with and without the demo header, to confirm
  environment-locking). Primary source for the EEA-host gotcha, the
  confirmed 0.20%/0.35% fee figures, and the sign-convention finding.
  Credentials used only in-memory for these calls, never written to any
  file.
- [50119 "API key doesn't exist" — freqtrade#10967](https://github.com/freqtrade/freqtrade/issues/10967),
  [ccxt#24601](https://github.com/ccxt/ccxt/issues/24601) — background on
  the error before the EEA-host cause was identified.
- [OKX now offers its services in the EEA through my.okx.com](https://www.okx.com/en-eu/help/okx-offers-its-services-in-the-eea-through-new-subdomain)
- [OKX Fees Guide 2026: Spot, Futures, Withdrawals](https://tradersunion.com/brokers/crypto/view/okex/fees/)
- [OKX fees: 0.020% maker, 0.050% taker (2026)](https://feeedge.com/exchanges/okx)
- [OKX API guide | OKX technical support](https://www.okx.com/docs-v5/en/)
- [OKX V5 Demo Trading API Guide for Traders](https://voiceofchain.com/academy/okx-v5-demo-trading-api)
- [OKX Demo Trading with Virtual Funds Guide](https://blockchain8.hashnode.dev/okx-demo-trading-virtual-funds-guide-2025)
- [okx - CCXT Docs](https://docs.ccxt.com/docs/exchanges/okx)
- [OKX X-Perps: How a 5-Year Expiry Clause Cracked Europe's Derivatives Market](https://blockeden.xyz/blog/2026/04/17/okx-x-perps-europe-mifid-5-year-expiry-perpetual-futures/)
- [OKX Restricted Countries List 2026](https://www.coinperps.com/learn/okx-restricted-countries)
- [Does OKX have a MiCA (CASP) license? Yes, licensed in Malta](https://casptracker.eu/exchange/okx/)
