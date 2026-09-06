# Saxo Bank

**Equities** (and forex/futures/options), not crypto — a second equities
candidate alongside Alpaca Europe, but a much more conventional shape:
a normal self-directed retail account with a real public API, not
Broker-as-a-Service. Saxo Bank A/S is Amsterdam-headquartered (Denmark
parent, EU-regulated, MiFID II). No Kairos account exists yet; this doc is
forward-looking, based on public docs only.

## The two headline findings

1. **Free, fully self-service demo API access — the best sandbox situation
   found across any candidate so far.** `developer.saxo/accounts/sim/signup`
   gives a free SIM account with a simulated €/$100,000 balance, no funded
   live account, no KYC needed to start. From there: create a Simulation
   Application in the Developer Portal, get an Application Key/Secret,
   pull a 24h OpenAPI access token, and the full OpenAPI surface (market
   data, cost precheck, order simulation) is usable immediately.
2. **Likely a real flat-floor problem, same shape as IBKR's.** Sources
   disagree on the exact number (see Fees below — this is the "sources
   disagree" case the exchange-docs README asks to flag rather than paper
   over), but every figure found has a **minimum commission per trade**,
   not pure percentage. On Kairos's ~€18 average trade, even the lowest
   found minimum (~€2 on Euronext ETFs) is over 10% round-trip; the
   higher stock-side figures (€10-12) would be worse than IBKR's $1 floor
   in percentage terms. **This needs a real `precheck` call against actual
   Kairos-sized orders before trusting either import, and it needs
   answering before assuming Saxo solves the problem IBKR had — it may
   not.**

## Auth & account setup

- SIM (demo) account: free, self-service, no funding — see finding 1 above.
- A **live** account needs normal retail KYC/funding, same as any broker.
  Not needed for Tier 1 discovery — the SIM environment can answer the cost
  questions this interface needs without one.
- Auth flow: OAuth2 (Application Key + Secret → access token). SIM tokens
  from the Developer Portal are short-lived (24h) for manual testing; a
  real integration would use the full OAuth2 code flow instead.

## Tier 1 interface mapping

| `ExchangeProbe` method | Saxo OpenAPI call |
|---|---|
| `connect(args)` | OAuth2 token exchange (Application Key/Secret → bearer token), SIM or live base URL |
| `resolve_instrument(symbol)` | `/ref/v1/instruments` (symbol/ISIN lookup → `Uic`, Saxo's internal instrument id, exchange, tick size) |
| `get_reference_price(instrument)` | `/trade/v1/infoprices` (live quote by `Uic`) |
| `get_cost_model(instrument, ref_price, base_currency)` | `POST /trade/v2/orders/precheck` with `"FieldGroups": ["Costs"]` — a non-transmitting order simulation returning `Commission`, `ExchangeFee`, `EstimatedTotalCost`, `EstimatedTotalCostInAccountCurrency` for the exact order you'd place |
| `get_fx_rate(currency, base)` | `EstimatedTotalCostInAccountCurrency` in the precheck response already includes FX conversion — a separate lookup likely isn't needed if account base currency is set correctly |

**Pattern is genuinely ambiguous, more A-like than B-like — worth flagging
explicitly since every other candidate so far has been cleanly one or the
other.** Saxo publishes a rate card (0.08% + a per-venue minimum) the way a
Pattern B exchange would, but getting the *exact* applicable number for a
specific order needs `/precheck` — a non-transmitting order simulation,
the same mechanism as IBKR's `whatIfOrder` (Pattern A). Unlike IBKR,
`/precheck` returns `Commission` directly in one call — no dual-probe
floor/rate inference needed, since Saxo's published schedule already tells
you the structure (percentage + minimum), you're just confirming the exact
number for one order. `infer_commission_model()` is not needed here.

## Fees — the critical check, and where sources disagree

No single authoritative public number was found; different sources (and
possibly different instrument sub-classes) give different minimums:

| Source/context | Rate | Minimum |
|---|---|---|
| US stocks, Classic tier | $0.02/share | $10 |
| European equities, Classic tier (one source) | 0.10% | €12 |
| European ETFs, Classic tier (another source) | 0.08% | €2 (Euronext) – €3 (Xetra), only above €2,500/trade |

Take none of these as settled — **run a real `/precheck` against a SIM
account for Kairos-sized ETF and equity orders before trusting any number
here.** Separately, Classic accounts also carry a **custody fee: 0.15%
p.a., minimum €5/month**, on stock/ETF/bond holdings — an ongoing holding
cost with no equivalent in any crypto exchange surveyed so far, worth
factoring in separately from per-trade commission.

## Order size / precision

Not confirmed from this pass — `/ref/v1/instruments` should expose lot
size/tick size per instrument (`InstrumentMeta.min_tick`/`size_increment`
in Kairos's schema); fractional-share support for the equities side was
not confirmed either way.

## Rate limits

Generous, not a sweep concern: **10,000,000 requests/day per application**
(all users/sessions combined), **120 requests/minute per session per
service group**, and a separate **1 order/second per session** cap (not
relevant to Tier 1 discovery, which never places a real order).

## Gotchas

- The Classic/Platinum/VIP tiering means the fee schedule genuinely
  changes with account balance — a number pulled from a SIM account may
  not represent what a real funded Classic account would see; confirm
  which tier a probe result reflects.
- Custody fee (above) is a cost type Kairos's `CostModel` schema doesn't
  currently have a field for (`commission_min`/`commission_rate_pct` are
  per-trade only) — worth a note if Saxo is ever actually implemented,
  not a blocker for a Tier 1 probe.
- `EstimatedTotalCost` vs `Commission` in the precheck response likely
  differ (total includes exchange fees, stamp duty, etc. beyond pure
  commission) — decide which one `CostModel.commission_min`/
  `commission_rate_pct` should actually track before implementing.

## What's needed before this can be probed for real

Nothing blocking except the decision to do it — per the free-SIM finding
above, this is genuinely one of the cheapest candidates to smoke-test:
sign up at `developer.saxo/accounts/sim/signup`, create a Simulation
Application, pull a token, and run real `/precheck` calls against
Kairos-representative order sizes to resolve the fee-figure disagreement
above. Not done this session (docs-only pass, 2026-09-06).

## Sources

- [Saxo Bank Developer Portal — Learn](https://www.developer.saxo/openapi/learn)
- [How do I get started with OpenAPI? – Saxo Bank Support](https://openapi.help.saxo/hc/en-us/articles/5231611647517-How-do-I-get-started-with-OpenAPI)
- [Create your OpenAPI Developer Account](https://www.developer.saxo/accounts/sim/signup)
- [How do I retrieve trade costs for orders? – Saxo Bank Support](https://openapi.help.saxo/hc/en-us/articles/4418459141265-How-do-I-retrieve-trade-costs-for-orders)
- [Saxo Bank Developer Portal — precheck endpoint reference](https://www.developer.saxo/openapi/referencedocs/trade/v2/orders/post__trade__precheck)
- [Saxo Bank Developer Portal — rate limiting](https://www.developer.saxo/openapi/learn/rate-limiting)
- [Saxo Fees: Full Breakdown for 2026 — BrokerChooser](https://brokerchooser.com/broker-reviews/saxo-bank-review/saxo-bank-fees)
- [Saxo Bank Fees (2026): Commission, Custody & FX Costs — QuantRoutine](https://quantroutine.com/brokers/saxo-bank-fees-explained/)
- [Saxo Bank Fees: Classic vs Platinum Breakdown — EntryLab](https://entrylab.io/brokers/saxo-bank/saxo-bank-minimum-deposit-fee-breakdown-classic-vs-platinum)
