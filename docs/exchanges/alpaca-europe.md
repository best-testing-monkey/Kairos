# Alpaca Europe

**Equities**, not crypto — the odd one out in this batch of 5. Alpaca
Europe A.V. is CNMV-regulated (Spain), launched April 2026 via acquiring
WealthKernel, MiFID II-compliant. Live equities coverage: **Xetra
(Germany) only** as of this research — Euronext (which includes Euronext
Amsterdam) and LSE are still listed as "coming," not live. No Kairos
account exists; this doc is forward-looking, based on public docs only.

## The load-bearing finding: this is Broker-as-a-Service, not a trading account

Alpaca has two distinct API products. The **Trading API** (US-only, as far
as these docs show) is a direct self-directed account — you sign up, you
trade your own money, paper trading included. The **Broker API** — which
is what `alpaca.markets/eu` and all EU documentation point to — is
different in kind: *"If you're a business integrating Alpaca as your
backend brokerage provider, you're using the Broker API."* It's built for
a fintech company to onboard **its own end users** (KYC-as-a-service,
ACATS transfers, ACH funding, journals, correspondent-level fee-setting)
and run a brokerage app on top of Alpaca's infrastructure — not for an
individual to open one account and trade it themselves.

Sign-up itself is described as free and self-service (a broker account +
API keys, a few minutes), so nothing technically stops Kairos from
registering as its own single "correspondent" and onboarding Baz as its
only "end user" — but architecturally this means Kairos would be acting as
the broker-of-record for itself, including **setting its own trading fee**
(see below) rather than just paying one. That's a materially heavier
integration than IBKR/Bitvavo/Kraken/Bybit, where you connect to an
already-existing account. Whether Alpaca's onboarding tolerates a
single-person "business" doing this is not established from public docs —
would need a real sign-up attempt to confirm.

## Fees

Alpaca's own stated policy: **no commission on trades**. But for Local
Currency Trading (LCT — trading US equities priced in EUR, relevant since
the account would be EUR-based) there are two separate charges: a
**correspondent fee** (which the broker-partner — i.e. whoever onboards as
Kairos would be — sets per order) plus a **fixed Alpaca fee** charged
regardless. The correspondent fee is quoted as an FX/swap fee in basis
points (e.g. 50bps on a JPY 10,000 trade = JPY 50). **Kairos would be
setting one of these two fees itself**, not just measuring what Alpaca
charges — a fundamentally different shape from every other exchange in
this survey, and worth flagging before any probe work starts here. No
public fee schedule was found for straight EUR-denominated Xetra trades
(as opposed to LCT specifically); would need the actual Broker API
onboarding flow to see a real number.

## Tier 1 interface mapping — unconfirmed, no working account to verify against

| Method | Likely backing call |
|---|---|
| `connect(args)` | OAuth2 Client Credentials flow — exchange API key/secret for a short-lived access token, then Bearer-auth every request |
| `resolve_instrument(symbol)` | Broker API asset/instrument lookup endpoint (exact path not confirmed — see `docs.alpaca.markets/eu/docs`) |
| `get_reference_price(instrument)` | Market data endpoint, shared with the US product's shape per the OpenAPI specs on `alpacahq/alpaca-docs` |
| `get_cost_model(instrument, ref_price, base_currency)` | Not a lookup at all here — see Fees above, this would mean reading back Kairos's *own* configured correspondent fee plus Alpaca's fixed fee, not discovering an externally-set one |
| `get_fx_rate(currency, base)` | LCT's swap-fee mechanism handles FX inline per trade; no separate spot-rate lookup was found documented |

This table is provisional — nobody has exercised these endpoints for this
project. Pattern A vs. B doesn't cleanly apply: there's no dry-run-order
probing (Pattern A) and no *externally-set* fee schedule to query directly
(Pattern B), because the correspondent fee is something Kairos would set
itself as the broker-partner.

## Order size / precision

Not confirmed for the EU entity. The US Trading API supports fractional
shares; whether Alpaca Europe's live Xetra offering does too is unconfirmed
from these docs.

## Rate limits

Not found documented for the EU Broker API specifically in this research
pass.

## Sandbox / testing

A sandbox environment exists for the (US) Broker API — described as a
parallel environment mirroring production prices/hours with no real
trades sent. Whether Alpaca Europe's entity has an equivalent sandbox is
not confirmed but plausible, since Broker API is one shared product
architecture across regions.

## Open questions / gotchas

1. **B2B-vs-direct-account ambiguity (see above) is the single biggest
   unresolved question.** Everything else in this doc is downstream of it.
2. **Euronext Amsterdam is not live.** Even if onboarding worked today,
   the NL-relevant venue for Dutch/EU equities isn't tradeable yet — only
   Xetra (German-listed instruments) is confirmed live.
3. Fee structure is circular for a self-onboarded single-user setup:
   Kairos would be both the fee-setter and the fee-payer, which makes
   "measuring the real cost" a different exercise than for every other
   exchange here (there is no external floor/rate to discover — it's a
   design decision).
4. No confirmation found of whether an individual (vs. a registered
   business entity) can complete Broker API onboarding at all.

## What's needed before this can be probed for real

Not started this session (scope: docs only). Before any code:
- Attempt actual sign-up at `alpaca.markets/eu/broker` to resolve the
  B2B-vs-individual question directly — this blocks everything else.
- If onboarding succeeds, get sandbox credentials and read the *actual*
  Broker API OpenAPI spec (`alpacahq/alpaca-docs` on GitHub) for exact
  endpoint paths, since this doc's Tier 1 mapping is inferred, not
  verified against a working integration.
- Re-check Euronext Amsterdam's live status before assuming NL equities
  are reachable at all.

## Sources

- [Modern Brokerage Infrastructure API for Europe - Alpaca](https://alpaca.markets/eu)
- [Getting Started with Broker API (EU docs)](https://docs.alpaca.markets/eu/docs/getting-started-with-broker-api)
- [Alpaca - Build Your Fintech App with Broker API](https://alpaca.markets/broker)
- [Getting Started with Broker API (US docs, same architecture)](https://docs.alpaca.markets/us/docs/getting-started-with-broker-api)
- [GitHub - api-evangelist/alpaca (OpenAPI spec index)](https://github.com/api-evangelist/alpaca)
- [Alpaca Launches Business Accounts for Broker Partners](https://alpaca.markets/blog/alpaca-launches-business-accounts-for-broker-partners/)
- [Alpaca Expands into Europe with WealthKernel Acquisition and Launch of European Equities Trading](https://www.businesswire.com/news/home/20260421441080/en/Alpaca-Expands-into-Europe-with-WealthKernel-Acquisition-and-Launch-of-European-Equities-Trading)
- [Alpaca Opens Access to German Stocks as Xetra Goes Live for Broker Partners](https://www.financemagnates.com/forex/alpaca-opens-access-to-german-stocks-as-xetra-goes-live-for-broker-partners/)
- [Local Currency Trading FAQ](https://alpaca.markets/support/local-currency-trading-faq)
- [Local Currency Trading (LCT) docs](https://docs.alpaca.markets/us/docs/local-currency-trading-lct)
- [Alpaca Support - Sandbox environment access](https://alpaca.markets/support/can-i-have-access-to-the-sandbox-environment)
