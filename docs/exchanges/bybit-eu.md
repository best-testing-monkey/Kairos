# Bybit EU

Bybit EU GmbH, Vienna — MiCA CASP-authorized by Austria's FMA (May 2025).
**Spot only for EU/NL retail**: spot trading, spot margin, Earn, Bybit
Card. Global Bybit's perpetuals/options are **not** offered on the EU
entity — crypto derivatives fall under MiFID II, not MiCA, and Bybit EU
hasn't launched them. Do not assume derivatives endpoints work here even
though they exist in the shared API surface.

**Spot margin is real, and now confirmed directly (Baz pasted the actual
page content 2026-09-06 after every automated fetch attempt failed — see
below) — and it's not a flat "leverage multiplier" the way the secondary
coverage framed it.** Bybit EU's UTA margin is a **collateralized-borrowing
system**: each of 42 supported coins has its own **borrow interest rate**
(quoted hourly and annualized, and separately as a "next" rate — these
update live) and its own **collateral ratio** — the haircut applied to that
asset's value when used as margin collateral, tiered *down* as your holding
size grows (a risk cap, not a single number). "Up to 10x leverage" in the
press coverage is a simplified headline over this mechanism, not a toggle
you set — effective leverage depends on which assets you hold/borrow and
how much.

Representative rates from the confirmed data (VIP0, all subject to
change — this is live data, not a static rate card):

| Coin | Borrowable | Annual borrow rate | Collateral ratio |
|---|---|---|---|
| USDT | Yes | ~2.28% | 100% (flat) |
| USDC | Yes | ~6.20% | 100% (flat) |
| BTC | Yes | ~1.25% | 98% (flat) |
| ETH | Yes | ~1.99% | 95% (flat) |
| SOL | Yes | ~1.34% | 90% (flat) |
| EUR | **No** | 0% (not borrowable) | 95% down to 0%, tiered by holding size (8 tiers, largest starting ~€4.3M) |

**EUR is margin-*eligible* (usable as collateral) but not
*borrowable*** — makes sense as the account's own base currency, and means
it's the natural collateral asset for an EUR-funded Kairos account rather
than something to borrow. Most collateral ratios are flat for major coins
(BTC/ETH/USDT/USDC/SOL/majors) but tier down in steps for higher-risk/
lower-liquidity coins as position size grows — e.g. a newer/smaller-cap
token might start at 80-85% collateral value and step down to 0% above a
specific quantity threshold, capping how much of a large holding can
actually back a margin position.

This confirms the speculative endpoint guess below (a genuine borrow-rate
lookup, separate from trading commission) was structurally right — the
real page is presenting exactly the two data points `init_margin_pct`/
a borrow-cost field would need: a per-coin rate and a per-coin (tiered)
ratio. Exact native endpoint path still unconfirmed (see below).

This is the one candidate in this directory where Kairos's
`CostModel.init_margin_pct`/`maint_margin_pct` fields would actually get
populated with something real for a crypto exchange — every other crypto
candidate surveyed (Bitvavo, Kraken, OKX, Bitstamp) is cash/spot only with
no margin concept at all.

**The announcement page itself could not be fetched by any automated
method** — `WebFetch` (HTTP 429, then repeated timeout) and a Playwright
headless-browser fetch (HTTP2 protocol error, then a full 30-60s timeout
even with HTTP2 disabled) both failed across 4 attempts, consistent with
active bot-blocking. Baz pasted the actual rendered content directly
instead, which is what the table and figures above are built from — this
is now first-party data, not secondary-source inference.

**USDT is not usable as a quote currency — CONFIRMED live (2026-09-13),
not just inferred from the MiCA rule.** Tether never sought MiCA
authorization, and a MiCA-licensed platform cannot offer non-authorized
stablecoins to EEA customers — so Bybit EU cannot run the USDT-quoted
markets that dominate global Bybit. A full `/v5/market/instruments-info`
sweep of all 131 spot pairs found **zero appearances of USDT anywhere**
(base or quote) — this is a hard regulatory absence, not a coverage gap
in how the universe/candidate checks were run. Bybit EU's only 3 quote
currencies are **EUR, PLN, USDC**; don't hardcode a `/USDT` suffix when
mapping symbols, verify the actual quote currency via `load_markets()`.
**PLN adds nothing beyond EUR/USDC**: every PLN-quoted base also has a
direct EUR and/or USDC pair (0 bases are PLN-only), and no `PLN/EUR`
pair exists on Bybit EU anyway, so there's no conversion route through
PLN even in principle — safe to ignore PLN entirely for coverage/candidate
purposes.

## Auth & account setup

- Testnet is free and separate from the live account: register directly
  at `testnet.bybit.com` (a different signup from bybit.com/bybit.eu).
  API-key creation is **blocked for the first 48h** after testnet
  registration (risk control). Once unblocked: Assets → Assets Overview →
  Request Test Coins gets 10,000 USDT + 1 BTC into the spot testnet
  account immediately, enough to probe spot market data and fee-rate
  endpoints without ever funding anything real.
- Testnet key management: `testnet.bybit.com/app/user/api-management`.
- No KYC needed to read public market data (symbols, tick sizes, market
  fee-rate defaults) — auth is only required for the account-specific
  `/v5/account/fee-rate` call (your actual negotiated rate) and anything
  order-related.
- Live Bybit EU account (when that's wanted) needs real KYC, separate
  from any global Bybit account.

## Tier 1 interface mapping (Pattern B)

| `ExchangeProbe` method | Bybit call |
|---|---|
| `connect(args)` | `ccxt.bybit({'apiKey': ..., 'secret': ...})` — public data needs no key |
| `resolve_instrument(symbol)` | `load_markets()[symbol]` (ccxt) or native `GET /v5/market/instruments-info` |
| `get_reference_price(instrument)` | `fetch_ticker(symbol)['last']` or native `GET /v5/market/tickers` |
| `get_cost_model(instrument, ...)` | `fetch_trading_fee(symbol)` (ccxt) or native `GET /v5/account/fee-rate` (needs auth — returns your actual `makerFeeRate`/`takerFeeRate`, not just the public VIP0 default) |
| `get_fx_rate(currency, base)` | `1.0` if quote currency == base_currency, else needs a separate spot pair lookup (e.g. quote-to-EUR) — Bybit EU being EUR-native for most pairs should make this mostly a no-op |

**If spot margin ever matters for `init_margin_pct`/`maint_margin_pct`**:
`fetch_trading_fee`/`fee-rate` above only covers *trading commission*, not
margin borrowing — confirmed now as a genuinely separate lookup (see the
real borrow-rate/collateral-ratio data above). Two calls would be needed,
not one: a per-coin borrow interest rate (likely native `GET /v5/
spot-margin-trade/interest-rate-history` or similar — exact path still
unconfirmed, the page itself couldn't be fetched to check the network
calls behind it) and a per-coin collateral-ratio/tier lookup (UTA
risk-limit endpoints). Not mapped to exact endpoints here — this session's
probe scope never needed margin math (every other crypto candidate has
none), so `get_cost_model`'s current mapping above stays trading-fee-only;
extending it to cover spot margin borrow cost is unstarted work with a
confirmed data shape, not a confirmed endpoint.

## The one real gotcha: EU endpoint vs. ccxt's default

**ccxt's `bybit` id points at `api.bybit.com` by default — not
`api.bybit.eu`.** As of this research, ccxt has no dedicated `bybit.eu` id
(see ccxt/ccxt#28665, open) — connecting to the EU-compliant, MiCA-scoped
endpoint requires manually overriding ccxt's URLs:

```python
exchange = ccxt.bybit({'apiKey': ..., 'secret': ...})
exchange.urls['api']['public'] = 'https://api.bybit.eu'
exchange.urls['api']['private'] = 'https://api.bybit.eu'
```

**CONFIRMED live (2026-09-13, plain HMAC-signed REST, no ccxt) — the two
domains are genuinely separate, not just a routing alias.** Public
`BTCEUR` tickers differ between them (66,497.6 on `.eu` vs. 66,328.9 on
`.com` at the same instant — separate order books/liquidity). More
importantly: **the EU-issued key is environment-locked** — authenticated
calls (`wallet-balance`, `fee-rate`) succeed on `api.bybit.eu`
(`retCode: 0`) and fail on `api.bybit.com` with `retCode: 10003, "API key
is invalid"`. Same safety property OKX/Bitstamp confirmed for their own
sandbox-vs-production splits: a key scoped to the compliant EU entity
cannot accidentally touch the wrong venue, even with Trade permissions.

## Fees

**CONFIRMED live (2026-09-13, real account, authenticated `/v5/account/
fee-rate` call for `BTCEUR`): 0.10% maker / 0.25% taker.** The pre-validation
guess of a flat 0.10%/0.10% below was wrong on the taker side — like every
other crypto candidate surveyed, maker/taker are asymmetric, not flat. No
IBKR-style flat-dollar floor (percentage only, confirmed not assumed). Fee
drops with 30-day volume (e.g. VIP3 quoted at 0.0625%/0.0750% on global
Bybit) — not reverified for the EU entity at higher tiers.

## Order size / precision

**CONFIRMED live (2026-09-13, public `/v5/market/instruments-info`,
`BTCEUR`)**: tick size 0.1 EUR, `minOrderQty` 0.00001 BTC, `minOrderAmt`
1 EUR (minimum notional). Per-symbol, not a global figure — reprobe for
other pairs before trusting these numbers elsewhere.

## Rate limits

600 requests / 5s per IP (120/s) by default — generous for a probe sweep,
no pacing needed like IBKR's 1.5s/request. Order-placement endpoints have
tighter limits (not relevant to Tier 1 discovery, which never places an
order). Response headers (`X-Bapi-Limit-Status`, `X-Bapi-Limit`) report
live remaining quota if it ever matters.

## What's needed before this can be probed for real

**Done — live-tested 2026-09-13.** ID verification (standard) approved
2026-09-06 23:32 UTC, same day as signup — much faster than the "up to 3
days" estimate reported at signup time. A system-generated (HMAC) API key
with Unified Trading/SPOT Trade + Convert permissions was created on the
real (non-testnet) account and validated with real read-only calls
(ticker, wallet-balance, fee-rate, instruments-info) — see "Fees", "Order
size / precision", and the endpoint-gotcha section above for the confirmed
figures. No order was placed despite the key being write-scoped (Trade
permission is for later live execution, not this Tier-1 discovery pass).
See `docs/universe-expansion-candidates.md`'s Bybit EU section for the
crypto-universe coverage measurement done in the same pass.

## Sources

- [Get Fee Rate | Bybit API Documentation](https://bybit-exchange.github.io/docs/v5/account/fee-rate)
- [Get Instruments Info | Bybit API Documentation](https://bybit-exchange.github.io/docs/v5/market/instrument)
- [Rate Limit Rules | Bybit API Documentation](https://bybit-exchange.github.io/docs/v5/rate-limit)
- [bybit - ccxt](https://docs.ccxt.com/exchanges/bybit)
- [Bybit.eu · Issue #28665 · ccxt/ccxt](https://github.com/ccxt/ccxt/issues/28665)
- [How to Register for a Testnet Account and Request for Test Coins](https://www.bybit.com/en/help-center/article/How-to-Request-Test-Coins-on-Testnet)
- [Bybit Trading Fee Structure - Help Center](https://www.bybit.com/en/help-center/article/Trading-Fee-Structure)
- [Bybit limits EEA access as MiCA deadline closes in](https://crypto.news/bybit-limits-eea-access-as-mica-deadline-closes-in/)
- [Does Bybit have a MiCA (CASP) license? Yes, licensed in Austria](https://casptracker.eu/exchange/bybit/)
- Bybit EU's own margin-data page (`bybit.eu/en-EU/announcement-info/fullstock-leverage-uta/`,
  flagged by Baz 2026-09-06) — **could not be fetched by any automated
  method** (WebFetch: HTTP 429 then repeated timeout; Playwright headless:
  HTTP2 protocol error, then timeout even with HTTP2 disabled — 4 attempts
  total, looks like active bot-blocking). **Baz pasted the actual rendered
  page content directly** (42-coin margin/collateral-ratio table, VIP0) —
  the borrow-rate and collateral-ratio figures above are built from that
  first-party data, not inference. Rates are live/dynamic; re-pull before
  trusting exact numbers on reuse.
- Secondary coverage (framed it as "up to 10x leverage" — directionally
  right as a headline, but the mechanism above is the accurate one):
  [Cointribune](https://www.cointribune.com/en/bybit-eu-launches-spot-margin-trading-with-leverage-up-to-10x/),
  [PRNewswire](https://www.prnewswire.com/news-releases/bybit-eu-empowers-european-traders-with-spot-margin-up-to-10x-leverage-full-transparency-and-built-in-risk-controls-302532221.html),
  [FinanceFeeds](https://financefeeds.com/bybit-rolls-out-spot-margin-trading-with-10x-leverage-under-mica-rules/),
  [CoinDesk](https://www.coindesk.com/business/2025/08/18/crypto-exchange-bybit-introduces-10x-spot-margin-trading-in-europe)
- [FAQ — Unified Trading Account (UTA), Bybit EU Help Center](https://www.bybit.eu/en-EU/help-center/article/FAQ-Unified-Trading-Account./)
