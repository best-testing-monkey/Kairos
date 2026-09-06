# Bybit EU

Bybit EU GmbH, Vienna — MiCA CASP-authorized by Austria's FMA (May 2025).
**Spot only for EU/NL retail**: spot trading, spot margin, Earn, Bybit
Card. Global Bybit's perpetuals/options are **not** offered on the EU
entity — crypto derivatives fall under MiFID II, not MiCA, and Bybit EU
hasn't launched them. Do not assume derivatives endpoints work here even
though they exist in the shared API surface.

**USDT is not usable as a quote currency.** Tether never sought MiCA
authorization, and a MiCA-licensed platform cannot offer non-authorized
stablecoins to EEA customers — so Bybit EU cannot run the USDT-quoted
markets that dominate global Bybit. Expect EUR- or MiCA-compliant-
stablecoin-quoted pairs instead; **don't hardcode a `/USDT` suffix when
mapping symbols**, verify the actual quote currency via `load_markets()`.

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

Untested by this doc (no account yet) — verify this override actually
routes correctly before trusting any data pulled through it; the two
domains may diverge in listed markets, not just legal entity.

## Fees

VIP0 (no volume) spot: **0.10% maker, 0.10% taker** — flat percentage,
no evidence of an IBKR-style flat-dollar floor (crypto exchanges generally
don't have one; confirm via a real `get_cost_model` probe rather than
trusting this doc). Fee drops with 30-day volume (e.g. VIP3 quoted at
0.0625%/0.0750% on global Bybit) — Bybit EU's own schedule may differ
slightly as a separate legal entity; the authoritative source is the
authenticated `/v5/account/fee-rate` call, not the public help-center page.

## Order size / precision

Not measured — `load_markets()`'s `precision`/`limits` fields
(`InstrumentMeta.min_tick`/`size_increment`/`min_size` in our schema) give
this per-symbol once probed. No global minimum-notional figure found in
research; check per-pair.

## Rate limits

600 requests / 5s per IP (120/s) by default — generous for a probe sweep,
no pacing needed like IBKR's 1.5s/request. Order-placement endpoints have
tighter limits (not relevant to Tier 1 discovery, which never places an
order). Response headers (`X-Bapi-Limit-Status`, `X-Bapi-Limit`) report
live remaining quota if it ever matters.

## What's needed before this can be probed for real

Nothing blocking except the decision to do it — this is the one candidate
where a real smoke test costs nothing: testnet signup, wait 48h for API-key
unblock, request test coins, probe away. Not done this session per Baz's
2026-09-06 scope call (scaffolding + playbook only); flagged here since
it's the cheapest of the four to revisit first if that changes.

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
