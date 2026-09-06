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
| [bybit-eu.md](bybit-eu.md) | crypto | Vienna, MiCA CASP | 0.10%/0.10% VIP0, spot only | No | B | Free public testnet — cheapest to smoke-test |
| [finst.md](finst.md) | crypto | Dutch, AFM | advertised 0.15%, unverified | Unknown | — | **No public API** — institutional-only, deprioritized |
| [alpaca-europe.md](alpaca-europe.md) | equities | Spain (CNMV), MiFID II | not externally set | N/A | — | **Broker-as-a-Service**, not a direct account — see doc |
| [saxo.md](saxo.md) | equities | Amsterdam-based (DK parent), MiFID II | disputed, €2-€12 min/trade (sources disagree — see doc) | **Likely yes**, same shape as IBKR | A (hybrid) | Free self-service SIM account, no KYC — best sandbox found |
| [okx.md](okx.md) | crypto | Malta, MFSA MiCA | 0.08%/0.10% | No | B | Free demo trading API, no funded account |
| [bitstamp.md](bitstamp.md) | crypto | Luxembourg, CSSF | 0.30%/0.40% | No (but €10 min order size) | B | Free sandbox, no funded account |

**Cheapest to actually smoke-test, ranked** (no funded account needed):
Bybit EU / OKX / Bitstamp (all free demo/sandbox, no wait) > Saxo (free SIM,
no wait, but equities not crypto) > Kraken/Bitvavo (need a real funded
account, no self-service sandbox found).

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

## Tone and rigor

- Dev-focused, not marketing copy. State uncertainty explicitly ("not
  confirmed", "sources disagree") instead of picking one number and
  presenting it as settled fact.
- No padding — match this repo's information-dense style, not a narrative
  writeup.
- It's fine, and expected, for a doc to say "unknown" or "not found in
  this pass" rather than guess.
