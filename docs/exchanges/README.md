# Exchange/broker connection docs

Implementation-ready references for connecting a specific exchange/broker
to Kairos's Tier 1 (`ExchangeProbe`) interface — see
[`docs/broker-api-interface.md`](../broker-api-interface.md) for what Tier
1 means and [`docs/playbooks/add-exchange-probe.md`](../playbooks/add-exchange-probe.md)
for how to actually build a probe script once a doc exists here.

All of these are **forward-looking**: no exchange in this directory has a
Kairos-owned account yet. Once one is actually probed for real, that
exchange's entry should move toward `docs/ibkr-cost-discovery.md`'s style
(empirical, measured) rather than staying research-only.

## Index

| Exchange | Class | NL/EU status | Base fee | Flat floor? | Pattern | API access |
|---|---|---|---|---|---|---|
| [bitvavo.md](bitvavo.md) | crypto | Dutch, DNB | 0.15%/0.25% | No | B | Public data open; fee-lookup likely needs a key |
| [kraken.md](kraken.md) | crypto | MiCA (Irish entity) | disputed, 0.16-0.80% (schedule changed Jul 2026 — verify live) | No | B | Public data open; real fee tier needs auth |
| [bybit-eu.md](bybit-eu.md) | crypto | Vienna, MiCA CASP | 0.10%/0.10% VIP0 trading fee; **plus** UTA margin — real per-coin borrow rates (BTC ~1.25%/yr, ETH ~1.99%/yr) + tiered collateral ratios, confirmed first-party | No (trading fee) | B | Free public testnet — cheapest to smoke-test |
| [finst.md](finst.md) | crypto | Dutch, AFM | advertised 0.15%, unverified | Unknown | — | **No public API** — institutional-only, deprioritized |
| [alpaca-europe.md](alpaca-europe.md) | equities | Spain (CNMV), MiFID II | not externally set | N/A | — | **Broker-as-a-Service**, not a direct account — see doc |
| [saxo.md](saxo.md) | equities | Amsterdam-based (DK parent), MiFID II | **CONFIRMED ~€12-13 flat** (live `precheck`, both a stock and an ETF) | **Yes — confirmed, worse than IBKR's $1** | A (hybrid) | Free self-service SIM account, no KYC — best sandbox found, but ruled out on fees |
| [okx.md](okx.md) | crypto | Malta, MFSA MiCA | **CONFIRMED 0.20%/0.35%** (EEA entity, live-tested — higher than global OKX's 0.08%/0.10%) | No | B | Free demo trading API — **live-tested and confirmed environment-locked (safe even with Trade/Withdraw perms)**. Needs `eea.okx.com`, not `www.okx.com` |
| [bitstamp.md](bitstamp.md) | crypto | Luxembourg, CSSF | **CONFIRMED 0.30%/0.40%** (live-tested, matches pre-measurement figure exactly) | No (but confirmed €10 min order size, live) | B | Free sandbox — **live-tested and confirmed environment-locked** (sandbox key rejected on production, `403 API key not found`) |

**Cheapest to actually smoke-test** (no *funded* account needed): Bybit EU
/ OKX / Bitstamp (free demo/sandbox) > Saxo (free SIM, equities not
crypto — **but already probed and ruled out on fees, see below**) >
Kraken/Bitvavo (need a real funded account, no self-service sandbox
found). **This is not the same as fastest to actually get access** — see
below, confirmed against real signup attempts 2026-09-06.

**Saxo is the first candidate actually probed for real, and it's a
no.** Live `precheck` calls (2026-09-06, SIM account) confirmed a flat
~€12-13 commission per trade regardless of instrument type — worse than
IBKR's $1 floor, not an improvement on it. See `docs/exchanges/saxo.md`.
No longer a live equities candidate unless something changes this
picture (e.g. a materially different tier).

**Bitstamp is the third candidate live-tested, and it's a pass on fees.**
Live sandbox calls (2026-09-06) confirmed 0.30%/0.40% fees (matching the
pre-measurement research exactly) and a €10 minimum order size — no flat
floor, same "percentage only" property as OKX. See
`docs/exchanges/bitstamp.md`.

**Real signup latency, first-hand (Baz, 2026-09-06):**
- **Bybit EU**: real ID verification gates even the testnet path, reported
  as up to **3 days**, *plus* the already-documented 48h API-key-unblock
  wait — up to ~5 days total before the API is actually usable. Contradicts
  this doc's earlier "free demo/sandbox, no wait" framing; that was about
  not needing a *funded* account, not about verification speed.
- **OKX**: reported as "5 minutes to prepare your account for identity
  verification" — sounds much faster to get moving on, though the full
  verification completion time (after that 5-minute prep step) isn't
  confirmed yet.
- **Bitstamp**: ID verification took up to 3 days — same latency class as
  Bybit EU, despite Bitstamp's "free sandbox, no funded account" framing
  above being about cost, not speed, same distinction already made for
  Bybit EU. Once cleared, sandbox key generation to live calls was
  same-day (like OKX/Saxo) — see `docs/exchanges/bitstamp.md`.
- **Saxo**: fast — same-day signup to live `precheck` calls (see
  `docs/exchanges/saxo.md`), no multi-day ID wait reported.

Every exchange in this table still needs a real probe run before any
number here is trustworthy — this index summarizes each doc's own
"critical fee check," it isn't itself a measurement.

## What every exchange doc must answer

Not rigid section headers — the existing docs vary in exact heading names —
but every doc in this directory must answer all of these, since a reader
comparing candidates needs to find each one without hunting:

1. **What is it, and is it actually usable from NL/EU right now?** Regulatory
   status (MiCA/AFM/DNB licensing for crypto, MiFID II for equities/CFDs) —
   briefly, this isn't a compliance memo, just enough to confirm it's a real
   candidate.
2. **Auth & account setup.** How you get API access, whether public data
   needs a key at all, and — this is the one gotcha that's bitten twice
   already (Finst, Alpaca) — whether "API access" actually means a normal
   self-service signup or something heavier (a sales conversation, a
   business-entity onboarding). Say so plainly if it's the latter.
3. **Tier 1 interface mapping.** A table: `ExchangeProbe` method → the real
   API/SDK call that backs it. State which cost-discovery pattern applies
   (Pattern A: dry-run/whatIf probing; Pattern B: direct fee-schedule
   lookup — see `docs/broker-api-interface.md`), or say plainly if neither
   fits cleanly (Alpaca Europe's doc is the precedent for that case).
4. **The critical fee check.** Is commission a flat floor per order (IBKR's
   problem — kills anything at Kairos's ~€18 average trade size) or pure
   percentage? State the actual base-tier number, not just "low fees" — and
   flag if sources disagree (Kraken's doc is the precedent: don't silently
   pick one number when the research turned up conflicting ones).
5. **Order size/precision quirks** — min notional, tick size, lot step.
6. **Rate limits.**
7. **Sandbox/testnet availability**, explicit yes/no, and roughly what it
   costs to get one (account needed? wait time? funded or free?). This is
   the signal that decides which candidate is cheapest to actually
   smoke-test first — Bybit EU's doc is the precedent for calling that out
   directly.
8. **Gotchas specific to this exchange** — anything that would trip up a
   first real implementation (Bybit's ccxt-endpoint mismatch, Kraken's
   public-vs-authenticated fee split).
9. **What's needed before this can be probed for real.** Explicit, since
   none of these have a Kairos-owned account. Don't imply "done" anywhere
   in the doc when nothing has actually been run against the exchange.
10. **Sources.** Every claim about fees/limits/API behavior needs one;
    list them at the bottom.

## When a finding changes the whole picture

Sometimes research surfaces something that isn't a "gotcha" but changes
whether the exchange is even the right *shape* of candidate — Finst having
no public API, Alpaca Europe turning out to be Broker-as-a-Service rather
than a direct account. When that happens, give it its own prominent section
near the **top** of the doc, before auth/fees/anything else — a reader
skimming should hit it first, not find it buried in a gotchas list six
sections down.

## Origin check (Israeli-ownership screen)

Baz excludes Israeli companies (Plus500 was ruled out on this basis, see
`project_broker_exchange_candidates` memory). Checked founder/origin for
all 8 candidates 2026-09-06 — **none are Israeli**:

| Exchange | Founders | Origin |
|---|---|---|
| Bitvavo | Tim Baardse, Jelle de Boer, Mark Nuvelstijn | Dutch |
| Kraken | Jesse Powell | American |
| Bybit EU | Ben Zhou | Chinese-born, HQ Singapore→Dubai |
| Finst | Julien Vallet, Marcel Putina, Maria Gallo (ex-DEGIRO team) | Dutch/European |
| Alpaca Europe | Yoshi Yokokawa, Hitoshi Harada | Japanese, HQ California |
| Saxo Bank | Kim Fournais, Lars Seier Christensen | Danish (2018: partial stake sold to Geely/China, Sampo/Finland) |
| OKX | Star Xu | Chinese, HQ now San Jose |
| Bitstamp | Nejc Kodrič, Damijan Merlak | Slovenian (now owned by Robinhood, US) |

Founder nationality is a reasonable proxy but not a legal ownership audit
— re-check if this criterion ever needs to be airtight rather than a
sanity check. Any *future* exchange added to this directory should get
the same check before being treated as a live candidate.

## Tone and rigor

- Dev-focused, not marketing copy. State uncertainty explicitly ("not
  confirmed", "sources disagree") instead of picking one number and
  presenting it as settled fact.
- No padding — match this repo's information-dense style, not a narrative
  writeup.
- It's fine, and expected, for a doc to say "unknown" or "not found in
  this pass" rather than guess.
