# Pulling real commission, margin and spread out of IBKR

Measured 2026-09-02 against a live IB Gateway **paper** account (DUQ322166,
base currency EUR, NetLiq ~€1M) on port 4002, via `ib_async`. Everything below
is empirical — probed, not read off a rate card. `scripts/ibkr_instruments.py`
implements it.

Nothing here places an order. Commission and margin come from `whatIfOrder`,
which IBKR evaluates and discards.

## Connecting

The project has no IB dependency; probes run with `--with ib_async` rather than
adding one:

```bash
UV_CACHE_DIR=/tmp/uv-cache-kairos uv run --with ib_async python -u <script>
```

`UV_CACHE_DIR` matters — `~/.cache` is bind-mounted onto the same NTFS/fuseblk
drive as the repo and uv's git checkout cache breaks there (see CLAUDE.md).

Ports: **4002** is the paper Gateway, 4001 live, 7497/7496 the TWS equivalents.
Use a distinct `clientId` per concurrent script or they evict each other.

## whatIf: the three rules that cost the most time

**1. Any warning collapses the result to `[]`.** `whatIfOrder` is typed as
returning `OrderState`, but ib_async resolves its future with an empty list the
moment an error or warning event arrives for that request. An off-market limit
price is enough. So is `Order TIF was set to DAY based on order preset` — the
warning you get by *not* setting `tif` explicitly. Always set `tif`, and always
use a limit price near the real market.

**2. Commission is only populated inside a notional band.** Roughly
**$249 – $65,200**. Outside it the field comes back UNSET with *no warning
emitted at all* — margin still returns fine, so a naive probe looks like it
worked. Measured boundaries:

```
F     5sh  $69       UNSET      AAPL 200sh  $65,200   OK
F    18sh  $249      OK         AAPL 250sh  $81,500   UNSET
AAPL  1sh  $326      OK         AAPL 400sh  $130,400  UNSET
```

This is not a Gateway precaution setting — an error listener attached across
the boundary captures nothing. Size probes into the band; don't try to widen
it. (`~/Jts/*/ibg.xml` is encrypted anyway, so Gateway settings can only be
changed through the GUI.)

**3. Limit prices must sit on the contract's tick grid.** Off-grid gives error
110 and, per rule 1, no commission. Round to `ContractDetails.minTick` —
BTC ticks at 0.25, EUR.USD at 0.00005, GC at 0.1.

Attach an error listener before probing. Without one every failure is an
indistinguishable `[]`; with one you get the actual reason, which is often
something structural rather than a bad request.

## What each field actually costs

| Want | Call | Level |
|---|---|---|
| Commission estimate | `whatIfOrder` → `OrderState.commission` | per contract + size |
| Real charged commission | `execDetails` → `commissionReport` | per fill (needs a real order) |
| Initial/maintenance margin | `whatIfOrder` → `initMarginChange` / `maintMarginChange` | per contract + size |
| Tick size, hours, size increment | `reqContractDetails` | per contract |
| Live spread | `reqMktData` ticks 1/2, `reqTickByTickData('BidAsk')` | per contract, needs subscription |
| Historical spread | `reqHistoricalData(whatToShow='BID_ASK')` | per contract, no subscription needed |
| Overnight financing | **not available** — see below | — |

For `BID_ASK` bars the semantics are non-obvious: **open = time-average bid,
close = time-average ask, high = max ask, low = min bid**. Average spread per
bar is therefore `close - open`. At 1-hour granularity this rounds to zero for
a liquid name (AAPL printed 326.09 for both); use 1-minute bars or
`reqHistoricalTicks('BID_ASK')`.

## Overnight financing is not in the socket API

Checked exhaustively, because phantom's `OvernightModel` wants it:

- A full field dump of `ContractDetails` for a stock CFD has **no** financing,
  interest or borrow field. The most cost-adjacent entry is
  `suggestedSizeIncrement`.
- ib_async exposes `shortable` (tick 46) and `shortableShares` (tick 89) —
  *availability only*. Grepping the library for borrow/rebate/fee-rate finds
  nothing. There is no fee-rate tick.
- `accountSummary` gives `AccruedCash` (€1,441.88) — one aggregate number for
  the whole account. Not per position, not a rate.

Financing must come from IBKR's published rate cards (benchmark ± spread) or
from Flex Queries / activity statements. `data/ibkr_shortable/*.csv` is a
symbol universe (`symbol,currency,long_name,country,yf_symbol`), **not** a rate
source — easy to mistake for one.

## Account-shaped constraints, not code bugs

These are properties of this account. They will differ on a live or
differently-permissioned one, and each one looks like a bug until identified.

**Segments.** Only two exist: `P = Crypto at Paxos` and `S = CFD`
(`TradingType-S = STKNOPT`). There is no commodities segment, so the futures
margin figures whatIf returns are theoretical — whatIf computes margin without
checking trading permission.

**Crypto cannot be whatIf'd, at any size.** The Paxos segment carries no margin
and is unfunded (all balances zero), so IBKR answers `MARGIN CALCULATION IS NOT
SUPPORTED FOR THIS CONTRACT`. Crypto also needs `tif='IOC'`, and `cashQty` only
works with market orders. Contract metadata still records fine. IBKR's Paxos
list is also far smaller than Kairos's 62-symbol crypto universe — XRP, DOGE,
ADA and AVAX have no contract at all.

**Some US ETFs are untradeable, per product.** EU PRIIPs/KID: *"This product
does not have a KID in English or in a language approved for your country."*
This is **not** predictable from domicile, venue or currency — SPY and DIA are
both US-listed ARCA ETFs in USD, and SPY is blocked while DIA trades fine.
Whether the issuer filed an approved-language KID for that specific product is
the discriminator, so availability has to be measured per symbol. Blocked in
the 2026-09-02 sweep: IWM, QQQ, SPY, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLU,
XLV, XLY.

ETFs also get materially better margin than single stocks — DIA 10.36% vs
AAPL 28.52% — which a single equity margin rate does not capture.

**Spot FX only prices in one direction.** The account holds EUR only, so buying
any pair shorts the quote currency and rejects with *"FX trade would expose
account to currency leverage"*. `SELL EURUSD` works; non-EUR pairs
(`GBPUSD`) reject in both directions.

**Fractional shares are desktop-only** (error 10243). Since `sizeIncrement`
reports 0.0001 for AAPL, sizing off it produces orders the API then refuses —
always round to whole units.

**No market-data subscriptions.** Delayed (`reqMarketDataType(3)`) works for
stocks; FX is live and free; crypto and futures return error 354/10089.

## Enumeration

`reqContractDetails` with no symbol **never returns** — CRYPTO/PAXOS,
CASH/IDEALPRO and CMDTY/SMART all hung past 100s. There is no wildcard dump.
Discovery has to go through `reqMatchingSymbols` (capped at 16 results per
query) or candidate lists probed by name, which is what
`scripts/ibkr_instruments.py --discover` does.

## Measured figures

Full sweep of `CANDIDATE_UNIVERSE` plus discovered instruments, 2026-09-02.
190 rows, 76 tradeable. Regenerate with
`uv run python scripts/ibkr_instruments.py --report`.

**Margin percentages are computed against a notional converted to the account
base currency.** `initMarginChange` is denominated in the base currency (EUR)
while the notional is in the contract's, so dividing one by the other directly
understates every USD figure by ~1.16x. That bug survived a full 153-symbol
sweep because 28.5% is exactly what US equity margin is *supposed* to look
like; a JPY-denominated CFD reading 0.02% is what finally exposed it. Every row
now stores `base_currency` and `fx_rate_to_base`, and `--report` warns if any
margin percentage lacks a rate.

### Commission

| Class | Formula |
|---|---|
| US stock / stock CFD | **min $1.00**, $0.005/share, cap 1% of value |
| Index CFD | ~0.005–0.06% of notional, min ~€0.5–0.93 |
| Metals (CFD and spot) | min ~€1.62 gold, €1.83 silver |
| Spot FX and FX CFD | 0.20 bp of trade value, **min €1.7252** |
| Futures | $2.36–$3.01 per contract |
| Crypto | unmeasurable — whatIf cannot price it |

Round-trip commission as a share of notional, US stocks. The floor is the whole
story at small size:

| Notional | Round trip |
|---|---|
| $277 | 0.72% |
| $326 | 0.61% |
| $652 | 0.31% |
| $1,630 | 0.12% |
| $3,378 | 0.06% |
| $65,212 | 0.00% |

`AllocationConfig.round_trip_cost_pct = 0.15` is accurate near $1,300 notional
and nowhere else. Kairos's measured average trade is ~€18, which is below the
fractional-share barrier and cannot be placed at all.

### Initial margin

| Instrument | Long | Short |
|---|---|---|
| US equity (40 symbols) | 32.82–35.70%, mean 33.11% | mean 35.61% |
| DIA (only tradeable ETF) | 12.01% | 12.01% |
| Index CFD (13) | 6.68–10.82% | mean 9.15% |
| Gold — XAUUSD CFD / CMDTY | 10.29% / 10.68% | same |
| Silver — XAGUSD CFD / CMDTY | 18.13% / 19.00% | 20.34% (CMDTY) |
| FX CFD | EUR 3.32, GBP 3.74, USD/JPY 3.31, AUD 5.00 | same |
| Spot FX | 0% (SELL direction) | mean 3.09% |
| Futures | ZS 7.24, ZC 7.59, GC 10.24, NG 13.70, CL 17.97, SI 19.20 | ~same |

Two things a single per-class rate cannot capture: **ETFs margin at roughly a
third of single stocks** (DIA 12.01% vs equity 33.11%), and **futures span
7–19%**, so oil and grain cannot share a rate.

**US stock CFDs carry the same margin as the cash share** — the CFD wrapper
buys no leverage here.

### What is not tradeable

| | n | Why |
|---|---|---|
| crypto | 62 | 54 have no PAXOS contract; 8 exist but cannot be margined |
| ETFs | 26 | PRIIPs/KID, per product |
| FX pairs | 21 | non-EUR pairs are leveraged from a EUR-only account |
| futures | 2 | HG=F, ZW=F returned no commission |

**50 of 153 universe symbols are tradeable on this account.** The tradeable
route to an index is an index CFD — `^GSPC`/`^IXIC`/`^DJI` resolve only as
`IND`, which is data-only.
