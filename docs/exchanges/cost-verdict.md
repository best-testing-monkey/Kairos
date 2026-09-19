# Broker cost verdict — who is cheap enough, at which order sizes?

Date: 2026-09-15. Question: does any screened broker/exchange let Kairos
trade its ~€18 average position within a sensible round-trip cost, and at
which sizes? Inputs are the measured figures in `docs/ibkr-cost-discovery.md`,
the confirmed-live fee figures in this directory's per-exchange docs, and the
coverage sweeps recorded in `docs/universe-expansion-candidates.md`. Nothing
below is new measurement — this is the cost arithmetic on already-measured
inputs.

## The benchmark, corrected first

The task brief says "we've been assuming 0.3% round-trip cost per position."
That number does not exist anywhere in the codebase or docs. What actually
exists:

- `strategy/allocation.py:61` — `round_trip_cost_pct: float = 0.15` (the
  value subtracted from every signal's shrunk EV in `compute_derived()`).
- Measured realized round-trip cost across all 539 trades of the recorded
  paper-trading run: **exactly 0.15% of notional** (spread €2.90 + slippage
  €1.93 + fx €9.67 + commission €0.00 over €9,665 notional — see
  `docs/papertrade_tickets/03-transaction-cost-accrual.md`). €9,665 / 539
  trades = **€17.9 average trade** — the ~€18 figure used throughout.
- `min_ev_pct` defaults: 0.10 (`kairos_signals.py:1507`), 0.15
  (`kairos_papertrade.py:1953`).

So the live assumption is **0.15%, not 0.3%**. This report evaluates both:
(a) the brief's 0.3% benchmark and (b) the stricter 0.15% the code actually
enforces. The composition of the measured 0.15% matters for what follows:
~0.05% was spread+slippage and ~0.10% was **fx conversion** — phantom traded
USD-quoted instruments from a EUR account. On a EUR-quoted crypto venue the
fx leg largely disappears, so a realistic all-in estimate for a EUR-native
crypto exchange is **commission round trip + ~0.05% slippage/spread**.

## Cost model

Round-trip commission on a pure-percentage venue is size-invariant; only a
flat floor amortizes. Two execution styles are shown because they differ by
2x on every venue: **taker×2** (market orders both sides — what Kairos's
current next-bar-open execution approximates) and **maker×2** (resting limit
orders both sides — achievable but unbuilt; no Kairos code path places limit
orders today).

Conversion hops are real trades with real fees. This bites one venue hard:
Bybit EU lists only EUR/USDC/PLN quotes (USDT is regulatorily absent), and
29 of its 41 covered symbols are **USDC-quoted only**, adding a USDC↔EUR hop
on both entry and exit (+0.50% taker round trip on top of the trade itself).

## Crypto venues — fees, coverage, effective round-trip cost

| Venue | Maker/Taker (base tier) | Confidence | Flat floor? | Min order | Conversion cost | Coverage of 62-symbol universe |
|---|---|---|---|---|---|---|
| **Bybit EU** | 0.10% / 0.25% | **Confirmed live** 2026-09-13 (real account, `/v5/account/fee-rate`) | No | €1 (BTC/EUR, confirmed) | None for 12 EUR-quoted symbols; **+0.50% taker RT** for 29 USDC-only symbols | **41/62 (66%)** |
| **OKX (EEA)** | 0.20% / 0.35% | **Confirmed live** 2026-09-06 (demo key, `eea.okx.com`) | No | not recorded (`minSz` field exists) | None — all 54 survivors have direct EUR pairs | **54/62 (87%)** |
| **Bitstamp** | 0.30% / 0.40% | **Confirmed live** 2026-09-06 (sandbox, `/api/v2/fees/trading/`) | No | **€10** (confirmed) | None except ZEC (USD hop) | **49/62 (79%)** |
| **Bitvavo** | 0.15% / 0.25% | Research only — **no account, no live test, no sandbox** | No (per published schedule) | per-market, unmeasured | None expected (EUR-native) | **Unmeasured** |
| **Kraken** | 0.16–0.40% / 0.26–0.80% (sources disagree; Jul 2026 schedule change) | **Unresolved — do not trust either end** | No (per published schedule) | per-pair, unmeasured | None expected (EUR pairs listed) | **Unmeasured** |

Effective round-trip cost (commission + ~0.05% slippage/spread, EUR-account
no-fx assumption) at the four order sizes. Because none of these venues has
a flat floor, the percentage is identical at every size — that row property
is the finding: **size does not rescue a percentage fee, and does not need
to**.

| Venue | Route / execution | €18 (base) | €36 (+100%) | €90 (+400%) | €180 (+900%) |
|---|---|---|---|---|---|
| Bybit EU | EUR pair, taker×2 | 0.55% | 0.55% | 0.55% | 0.55% |
| Bybit EU | EUR pair, maker×2 | **0.25%** | **0.25%** | **0.25%** | **0.25%** |
| Bybit EU | USDC-routed, taker×2 | 1.05% | 1.05% | 1.05% | 1.05% |
| Bybit EU | USDC-routed, maker×2 | 0.45% | 0.45% | 0.45% | 0.45% |
| OKX | taker×2 | 0.75% | 0.75% | 0.75% | 0.75% |
| OKX | maker×2 | 0.45% | 0.45% | 0.45% | 0.45% |
| Bitstamp | taker×2 | 0.85% | 0.85% | 0.85% | 0.85% |
| Bitstamp | maker×2 | 0.65% | 0.65% | 0.65% | 0.65% |
| Bitvavo (unverified) | taker×2 | 0.55% | 0.55% | 0.55% | 0.55% |
| Bitvavo (unverified) | maker×2 | 0.35% | 0.35% | 0.35% | 0.35% |
| Kraken (unresolved) | taker×2 | 0.57%–1.65% | same | same | same |

Contrast with the flat-floor brokers, where size is the whole story
(commission round trip only, equities venues — shown to make the amortization
contrast explicit):

| Broker | €18 | €36 | €90 | €180 |
|---|---|---|---|---|
| IBKR (min $1/side, confirmed 2026-09-02) | ~11% (and unplaceable — below fractional-share API barrier) | ~5.5% | ~2.2% | ~1.1% |
| Saxo (~€12.5/side, confirmed 2026-09-06 `precheck`) | ~139% | ~69% | ~28% | ~14% |

Both remain untradeable even at €180. Equities at Kairos sizes have no viable
candidate among the eight screened (Finst: no public API; Alpaca Europe:
Broker-as-a-Service, not an account; Saxo/IBKR: flat floors).

## Pass/fail

**(a) Against the 0.3% benchmark (all-in round trip):**

| Venue | taker×2 | maker×2 |
|---|---|---|
| Bybit EU (EUR pairs) | FAIL (0.55%) | **PASS (0.25%)** |
| Bybit EU (USDC-routed) | FAIL (1.05%) | FAIL (0.45%) |
| OKX | FAIL (0.75%) | FAIL (0.45%) |
| Bitstamp | FAIL (0.85%) | FAIL (0.65%) |
| Bitvavo | FAIL (0.55%) | borderline FAIL (0.35%) — unverified anyway |
| Kraken | indeterminate | indeterminate |

**(b) Against the code's actual 0.15% cost allowance:** nothing passes, on
any venue, in any execution style. The cheapest confirmed option (Bybit EU
maker×2 at 0.25% all-in) is already 1.7x the allowance before slippage
variance. **The 0.15% assumption is not achievable on any EU-regulated spot
crypto venue surveyed; if it is load-bearing for signal selection, it should
be re-justified or raised** — see `docs/papertrade_tickets/03-transaction-
cost-accrual.md`, which already flags per-asset-class `round_trip_cost_pct`
as open.

**(c) Against measured per-trade EV (break-even):** the two realized
measurements bracket zero. Run 1 (539 trades, accounting-bug-corrected):
**+0.57% mean return per trade**; the 2026-07-26 rerun (423 trades):
**−0.26% per trade** (`docs/papertrade_loss_analysis.md`). Break-even total
cost therefore sits somewhere between −0.26% and +0.57% — i.e. under the
optimistic reading only ~0.5% of headroom exists for commission + slippage,
and under the pessimistic reading none at all. At taker execution **no
surveyed venue fits inside even the optimistic break-even**; only Bybit EU
maker×2 on EUR pairs (0.25%) and, barely, Bybit EU taker×2 on EUR pairs
(0.55% ≈ the +0.57% edge, leaving ~zero net) fit. Note the selection
pipeline already demands projected `ev_net > 0` after a 0.15% cost haircut,
so live-selected signals carry higher *projected* EV than the realized mean
— but projected EV is exactly what the two realized runs failed to deliver.

## Ranked recommendation

1. **Bybit EU — pass, conditionally.** Cheapest confirmed fees
   (0.10%/0.25%), €1 minimum order, environment-locked EU keys, and the only
   venue that beats 0.3% (maker execution on EUR pairs) or touches the
   optimistic EV break-even (taker, EUR pairs). Two real costs: worst
   coverage of the tested three (41/62, 66%), and 29 of those 41 symbols are
   USDC-routed at an effective 1.05% taker round trip — for those symbols
   Bybit EU is the *most* expensive tested venue, not the cheapest. It is
   also the only candidate with real spot margin (per-coin borrow rates,
   BTC ~1.25%/yr) if leverage ever matters. Practical scope: EUR-quoted
   majors only, unless maker execution is built.
2. **Bitvavo — probe next, could displace #1.** Published 0.15%/0.25% with
   no floor, EUR-native (no conversion hops at all), Dutch-regulated. If a
   live `fetchTradingFee` confirms the schedule and coverage comes back
   reasonable, it matches Bybit EU's taker cost with no USDC surcharge and
   likely broader EUR coverage. But today every number is unverified, there
   is no sandbox, and coverage against the 62-symbol universe has never been
   measured — cheapest next measurement in the whole program (one account +
   one authenticated call + one public instruments sweep).
3. **OKX — the coverage/fee compromise, but not cheap enough at base
   tier.** Best confirmed coverage (54/62, 87%), all survivors direct-EUR
   (no hop surcharge), confirmed 0.20%/0.35%, excellent demo API. Fails both
   cost benchmarks at base tier; the case for it rests on whether OKX EEA's
   VIP tiers drop with volume like global OKX's (unconfirmed — and Kairos's
   volume will not reach meaningful tiers at €18–€180 trades).
4. **Bitstamp — pass on structure, fail on price.** No floor, clean Pattern
   B API, honest €10 minimum (which €18 clears by only 1.8x — small signals
   will be blocked outright, check the signal-size distribution before
   committing). But 0.30%/0.40% is the most expensive confirmed schedule and
   coverage (79%) is second-best. Dominated by OKX on both axes.
5. **Kraken — unmeasurable, therefore unrankable.** Fee sources span
   0.16%–0.80% taker after the July 2026 schedule change; no spot sandbox;
   no account. One authenticated `TradeVolume` call resolves it — until then
   no verdict is defensible.

**Not viable at any tested size:** IBKR, Saxo (flat floors — 1.1% and 14%
respectively even at €180), Finst (no API), Alpaca Europe (BaaS, not an
account). Equities have no candidate; crypto does.

## What this does not settle

- Maker execution is a code change, not a config flag — nothing in Kairos or
  phantom places resting limit orders today, and the whole "pass" column
  above depends on it (plus fill-risk analysis nobody has done).
- Bitvavo and Kraken coverage of the 62-symbol universe is unmeasured.
- The 0.15% vs 0.3% assumption discrepancy (top of this doc) is a real open
  question for `AllocationConfig` — the achievable-cost evidence here
  supports raising it, not lowering it.
- All fee figures are base-tier. None of Kairos's projected volumes reach
  any venue's VIP tiers, so no tier upside was assumed anywhere above.

## Sources

- `docs/ibkr-cost-discovery.md` — IBKR measured commission floor, margin,
  tradeability (2026-09-02).
- `docs/exchanges/{bybit-eu,okx,bitstamp}.md` — confirmed-live fee/minimum/
  coverage figures (2026-09-06 / 2026-09-13).
- `docs/exchanges/{bitvavo,kraken,finst,alpaca-europe,saxo}.md` — research-
  only or ruled-out candidates.
- `docs/universe-expansion-candidates.md` — 62-symbol coverage sweeps (OKX
  87%, Bitstamp 79%, Bybit EU 66%).
- `docs/papertrade_tickets/03-transaction-cost-accrual.md` and
  `docs/papertrade_loss_analysis.md` — measured realized cost (0.15%),
  ~€18 average trade derivation, per-trade EV measurements.
- `strategy/allocation.py:61` — `round_trip_cost_pct = 0.15`.
