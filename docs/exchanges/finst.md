# Finst

Dutch crypto platform, AFM-licensed (MiCAR) since July 2025, DNB-registered.
340-400+ assets, flat 0.15% fee advertised, custody via Fireblocks, fiat via
bunq/ING. Amsterdam-based.

## Auth & account setup

**No public developer portal or self-serve API docs found.** Finst's site
mentions API access ("a powerful suite of tools via one API") only under
its **Institutional** and **Corporate** product pages — no `developers.
finst.com`/`docs.finst.com`, no public endpoint reference, no SDK. The only
route in is contacting `institutional@finst.com` directly. This is a
materially different access model from Bitvavo/Kraken/Bybit, which all
publish open API docs and let you generate a retail API key from account
settings with no sales conversation required.

**Not in ccxt.** Checked `docs.ccxt.com/docs/exchange-markets` (104
supported crypto exchanges) — Finst is absent.

## Tier 1 interface mapping

Not possible to fill in — there is nothing publicly documented to map
`resolve_instrument`/`get_reference_price`/`get_cost_model`/`get_fx_rate`
against. Would require getting Finst's actual API reference from their
institutional team first.

## Fee schedule specifics

Advertised: flat **0.15%**, spreads "as low as 0%". No minimum-trade-amount
or flat-floor caveat surfaced anywhere in what's public — but this is
retail-page marketing copy, not a rate card, and it's unclear whether
institutional/API-tier pricing matches the retail 0.15% or differs. Cannot
confirm there's no IBKR-style floor without the actual institutional
agreement.

## Order size/precision quirks

Unknown — no public API reference to check against.

## Rate limits

Unknown.

## Known gotchas

- The API being institutional/corporate-gated (not self-serve) is itself
  the gotcha: this isn't a "sign up, generate a key" flow like the other
  three crypto candidates — it likely means a sales conversation, possibly
  a minimum relationship size, before any API access exists at all.
- Everything published is retail marketing content; no rate-card or API
  reference document was found to verify fee/precision claims against.

## What's needed before this can be probed for real

Not started this session (scaffolding-only pass, 2026-09-06). Before
Finst can even get a `docs/exchanges/finst.md` interface mapping filled in,
someone needs to actually contact `institutional@finst.com` and get real
API documentation — this is a different, heavier first step than
Bitvavo/Kraken/Bybit EU, which just need an account signup. Given that gap,
**Finst should probably be deprioritized relative to the other three
crypto candidates** unless the 0.15% flat-fee figure specifically (vs.
Kraken's 0.40% base tier) is worth the extra friction to confirm.

## Sources

- [Finst Institutional](https://finst.com/en/institutional)
- [Finst — About](https://finst.com/en/company)
- [CCXT — Supported Exchanges](https://docs.ccxt.com/docs/exchange-markets)
- [Finst Exchange 2026 overview — TradingFinder](https://tradingfinder.com/exchanges/finst/)
