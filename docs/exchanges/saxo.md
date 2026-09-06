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
2. **CONFIRMED (2026-09-06, live SIM `precheck` calls, not research):
   Saxo has a real flat-per-trade commission floor, and it is *worse* than
   IBKR's, not better.** Baz signed up for the free SIM account, generated
   a 24h token, and two live `POST /trade/v2/orders/precheck` calls were
   run against it:
   - 1 share of AAPL (Uic 211, NASDAQ): **Commission $15.00 / €12.92**
     (`CostInAccountCurrency`, `PreCheckResult: "Ok"`).
   - 1 unit of IWDA (Uic 50629, iShares Core MSCI World UCITS ETF,
     Euronext Amsterdam, EUR-denominated): **Commission €12.00**
     (`PreCheckResult: "Ok"`).

   Both land in the same ~€12-13 range regardless of asset type (stock vs.
   ETF) or order size (1 share/unit in both cases) — this reads as Saxo's
   Classic-tier **minimum commission per trade**, not a percentage that
   happened to be small. On Kairos's ~€18 average trade, a ~€12-13 flat
   commission is **65-70% of the trade's entire value round-trip on one
   leg alone** — categorically worse than IBKR's $1 floor (which was
   "only" ~0.6-0.7% at a placeable size). **Saxo does not solve the
   problem IBKR had. It is currently the single worst-fee candidate
   surveyed across all 8 exchanges/brokers**, worse even than Bitpanda's
   1.00-1.49% (deprioritized crypto candidate, see
   `docs/exchanges/README.md`).

   This was 2 data points, not a full sweep (no `scripts/saxo_instruments.py`
   exists yet — see `docs/playbooks/add-exchange-probe.md`), but both
   landed in the same tight range on different asset classes, which is
   fairly strong evidence this is a real floor, not noise. The earlier
   "€2-3 Euronext ETF" secondary-source figure did not hold up against a
   real order — either it applies only above some higher notional this
   1-unit order didn't reach, or it was simply wrong; not disambiguated.

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

## Fees — CONFIRMED 2026-09-06 via live SIM `precheck` calls

| Instrument | Order | Commission (account currency) | `PreCheckResult` |
|---|---|---|---|
| AAPL (Uic 211, NASDAQ) | 1 share, market buy | **€12.92** ($15.00 instrument ccy) | Ok |
| IWDA (Uic 50629, Euronext Amsterdam ETF) | 1 unit, market buy | **€12.00** | Ok |

Both land in the same ~€12-13 range — reads as a **flat minimum commission
per trade**, not a percentage. The pre-session research table below is
kept for context but **should not be trusted over the measured numbers
above** — none of the researched figures (a $10 US-stock minimum, a €2-3
Euronext-ETF minimum) held up against a real order:

| Source/context (pre-measurement research, unreliable) | Rate | Minimum |
|---|---|---|
| US stocks, Classic tier | $0.02/share | $10 |
| European equities, Classic tier (one source) | 0.10% | €12 |
| European ETFs, Classic tier (another source) | 0.08% | €2 (Euronext) – €3 (Xetra), only above €2,500/trade |

Only 2 data points measured (not a full sweep — no `scripts/saxo_instruments.py`
exists yet), but landing in the same tight range across two different
asset classes (stock vs. ETF) is fairly strong evidence this is real, not
noise. Separately, Classic accounts also carry a **custody fee: 0.15%
p.a., minimum €5/month** (unmeasured, from research only), on stock/ETF/
bond holdings — an ongoing holding cost with no equivalent in any crypto
exchange surveyed so far, worth factoring in separately from per-trade
commission if Saxo is ever reconsidered despite the commission finding.

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

- **`get_reference_price` (`/trade/v1/infoprices`) returned `200 Ok` but
  empty `Bid`/`Ask` fields** on a fresh SIM account, confirmed 2026-09-06.
  The `/port/v1/users/me` response includes
  `"MarketDataViaOpenApiTermsAccepted": false` — almost certainly the
  cause (an explicit terms-acceptance step, gating live/delayed quotes,
  separate from the account/API being otherwise fully functional — auth,
  instrument search, and cost `precheck` all worked cleanly). Not yet
  confirmed where that acceptance happens (likely somewhere in the
  Developer Portal or a dedicated endpoint) — flagged for whoever picks
  this up next rather than guessed at.
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

## What's been probed for real, and what's still open

**Done 2026-09-06**: signup, Simulation Application, 24h token, and 3 live
calls — `/port/v1/users/me` (auth check, Ok), `/ref/v1/instruments`
(symbol search, Ok), and 2× `/trade/v2/orders/precheck` (the critical fee
check, both confirmed a ~€12-13 flat commission — see Fees above). This
was enough to settle the headline question (does Saxo solve IBKR's
flat-floor problem — no) without building a full probe script.

**Still open**: no `scripts/saxo_instruments.py` exists (would formalize
this into the shared `ExchangeProbe` framework per
`docs/playbooks/add-exchange-probe.md`, useful mainly if Saxo gets
reconsidered later despite the fee finding); the market-data terms gotcha
above is unresolved; only 2 instruments were checked, both landing in the
same range but not a full sweep; custody fee (0.15% p.a.) unmeasured.
Given the confirmed fee finding, further Saxo work is probably lower
priority than the other 7 candidates unless something changes that
picture (e.g. a higher account tier with a materially different
commission schedule).

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
