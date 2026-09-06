# Connecting Kraken (Tier 1 — cost/tradeability discovery)

Global crypto exchange, MiCA-authorized via its Irish entity for EU/NL
retail. No Kairos account exists yet — this is how to connect it, not a
measured result (see `docs/broker-api-interface.md` for the Tier 1/Tier 2
split, `docs/ibkr-cost-discovery.md` for the empirical-measurement
precedent this doc is meant to set up, not duplicate).

## Auth & account setup

- API key: Settings → API → Add key, on a verified Kraken account. Scope
  narrowly for a probe: `Query Funds`, `Query Open Orders & Trades` is
  enough; do **not** grant `Create & Modify Orders` for a read-only cost
  sweep.
- Public endpoints (`AssetPairs`, `Ticker`) need no key at all.
- The *personalized* fee-tier endpoint (`TradeVolume`, what ccxt's
  `fetchTradingFee` calls under the hood for your actual rate) requires
  authentication — the fee returned without auth is the published base-tier
  schedule, not necessarily what your account would actually pay.
- **No self-service spot sandbox.** Kraken's Futures API has a public
  sandbox (`demo-futures.kraken.com`), but Spot UAT is provisioned only on
  request through an Account Manager — not something you can self-serve
  from a dev account. Any real spot probe therefore runs against
  production, on public/read-only endpoints only.

## Tier 1 interface mapping (Pattern B — direct fee-schedule lookup)

| `ExchangeProbe` method | Kraken/ccxt call | Auth needed |
|---|---|---|
| `resolve_instrument(symbol)` | ccxt `load_markets()` / `market(symbol)` (native: `AssetPairs`) | No |
| `get_reference_price(instrument)` | ccxt `fetch_ticker(symbol)` (native: `Ticker`) | No |
| `get_cost_model(instrument, ref_price, base)` | ccxt `fetch_trading_fee(symbol)` (native: `TradeVolume`) | Yes, for your real tier — public call returns only the published base schedule |
| `get_fx_rate(currency, base)` | Not generally needed — Kraken lists both crypto and fiat pairs directly (e.g. `BTC/EUR`); only relevant if quoting in a currency your account base currency isn't | — |

Confirmed Pattern B: no dry-run/whatIf order needed, `infer_commission_model`
is not used here — one `fetch_trading_fee` call is the whole cost lookup.

## Fee schedule

**No flat minimum commission per order** — fees are purely percentage of
notional, which is the property that ruled out IBKR for Kairos's ~€18 avg
trade (IBKR's flat $1 floor was 5-7% of that trade alone).

Base-tier taker/maker %: sources disagree in a way that looks like it's
tracking Kraken's **July 9, 2026 fee-schedule change**, not a real
ambiguity — figures found range from 0.80%/0.40% down to 0.26%/0.16%
taker/maker depending on which cached page a source pulled from. Tiers are
now based on the **best of** three metrics: 30-day spot volume, 30-day
futures volume, or Assets on Platform (AoP) — so an account can qualify for
a lower tier by balance alone, not just turnover. **Don't trust either
number above for a real decision** — pull the live schedule from
`kraken.com/features/fee-schedule` or an authenticated `TradeVolume` call
before it matters, the same discipline `docs/ibkr-cost-discovery.md`
enforced by probing rather than reading a rate card.

## Order size/precision quirks

`AssetPairs` (ccxt: `market['limits']['amount']['min']`, Kraken's own field
name is `ordermin`) gives per-pair minimum order size; `market['precision']`
gives tick/lot precision. Not measured here — pull it per-pair when
actually probing, same as IBKR's `min_tick`/`size_increment` fields.

## Rate limits

Counter-based per API key, decremented over time at a rate that depends on
your verification tier; a **separate counter per currency pair**, so
hitting the limit on one pair doesn't block others. Exact call-weight
figures weren't nailed down in this pass — check
`docs.kraken.com/api/docs/guides/spot-rest-ratelimits/` before writing a
sweep loop that fires many requests per second.

## Gotchas

- The public vs. authenticated fee-tier distinction above is easy to miss
  and would silently under- or over-state real cost if you probe without a
  key and assume the answer is your actual rate.
- No spot sandbox means any real probe touches production data — read-only
  scope keeps that safe, but there is no dry-run equivalent to IBKR's
  `whatIfOrder` here; you're not placing orders either way (Pattern B
  doesn't need to), so this is lower-stakes than it sounds, just worth
  knowing going in.

## Not done this session

No Kairos-owned Kraken account or API key exists. Per Baz's 2026-09-06
decision, this pass built the framework and this connection doc only —
scaffolding + playbook, not a live sandbox/production smoke test. Next step
per `docs/playbooks/add-exchange-probe.md`: create an account, generate a
read-only key, implement `scripts/kraken_instruments.py` against the
mapping above, run it.

## Sources

- [Quickstart - Kraken Developers](https://docs.kraken.com/home/guides/quickstart)
- [API Testing Environment | Kraken](https://support.kraken.com/articles/360024809011-api-testing-environment-derivatives)
- [Fee Structures | Kraken](https://www.kraken.com/features/fee-schedule)
- [Cross-platform fee tier changes (July 2026) | Kraken](https://support.kraken.com/articles/cross-platform-fee-tier-changes)
- [docs.ccxt.com/docs/exchanges/kraken](https://docs.ccxt.com/docs/exchanges/kraken)
- [Spot Trading Limits - Kraken Developers](https://docs.kraken.com/api/docs/guides/spot-ratelimits/)
