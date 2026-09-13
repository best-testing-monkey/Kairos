# Universe expansion candidates

Candidates for `strategy/kairos_pipeline.py`'s `CANDIDATE_UNIVERSE`,
sourced from real exchange listings as broker/exchange connections get
validated (see `docs/exchanges/`). Directly serves the open thread in
`project_crypto_commodity_universe_expansion` memory — the deduped corpus
is ~95% equity (916 equity / 29 crypto / 4 fx_commodity), which starves
any per-class comparison; more crypto/commodity coverage is the fix.

**Nothing here is in `CANDIDATE_UNIVERSE` yet.** This is a discovery list,
not an addition — per that memory's own instruction, the order is universe
screening → correlation/grouping → oracle + naive → base → small/mini →
regenerate `docs/papers/`, and none of that has run against these symbols.
Adding a symbol to `CANDIDATE_UNIVERSE` is a separate, deliberate step
after real screening, not automatic from appearing here.

## Method

For each connected exchange: pull the full public instrument + 24h-volume
listing (one or two public API calls, no auth needed), map base tickers
against yfinance-style symbols already in `CANDIDATE_UNIVERSE["crypto"]`
(stripping yfinance's numeric disambiguation suffixes, e.g.
`POL28321-USD` → `POL`), exclude stablecoins (not useful for directional
prediction), and keep anything at or above Kairos's own current
lowest-volume crypto entry — a real, already-accepted floor, not an
arbitrary cutoff.

## OKX (2026-09-06)

Floor: JTO-EUR's ~8,139 EUR/24h volume (Kairos's current lowest-volume
crypto entry). 213 non-stablecoin OKX-listed assets aren't in Kairos's
universe at all; **38 clear the floor**:

| Symbol | 24h EUR volume | Notes |
|---|---|---|
| HYPE | 829,297 | Hyperliquid |
| PUMP | 633,421 | |
| SUSHI | 262,814 | SushiSwap, established DeFi |
| BOME | 229,406 | |
| ENA | 171,507 | Ethena |
| OKB | 139,491 | OKX's own exchange token — consider excluding, exchange-specific bias |
| TAO | 130,363 | Bittensor |
| ZEN | 126,938 | Horizen |
| TRUMP | 74,253 | political meme coin |
| ASTER | 71,878 | |
| MON | 71,089 | |
| ORDI | 61,658 | Ordinals |
| GRVT | 50,725 | |
| PENGU | 44,500 | Pudgy Penguins |
| RE | 36,572 | |
| MOODENG | 36,415 | |
| IOST | 33,581 | |
| ICX | 26,699 | ICON |
| XPL | 26,290 | |
| ZAMA | 26,216 | |
| RAY | 24,885 | Raydium, established Solana DeFi |
| FET | 20,013 | Fetch.ai, established AI-crypto |
| PI | 19,726 | |
| CATI | 18,959 | |
| VIRTUAL | 17,518 | Virtuals Protocol, notable AI-agent token |
| HUMA | 14,886 | |
| PENDLE | 13,809 | established DeFi |
| NIGHT | 13,482 | |
| LPT | 13,158 | Livepeer |
| PEOPLE | 12,737 | |
| OL | 12,012 | |
| TRB | 11,669 | Tellor |
| **TRX** | 9,882 | **TRON — top-15 market cap, worth a sanity check on why it's not already in the universe** |
| CORE | 9,772 | |
| STX | 9,360 | Stacks |
| SPK | 9,292 | |
| FOGO | 8,883 | |
| NMR | 8,496 | Numeraire |

Excluded as stablecoins (not directional-prediction candidates):
USDC-EUR (3.5M vol), USDT-EUR (2.5M vol), USDG-EUR (29.9k vol).

## Bitstamp (2026-09-06)

Checked Kairos's full 62-symbol crypto universe against Bitstamp's public
`GET /api/v2/trading-pairs-info/` (257 pairs total, 112 EUR-quoted — one
call covers everything, no per-symbol lookup needed).

- **49/62 (79%) are tradeable against EUR, directly or via free
  conversion** — 47 have a direct EUR pair; 1 via a legacy ticker
  (`RENDER-USD` is listed on Bitstamp as `RNDR/EUR`,
  `"description": "Render Token / Euro"` — Bitstamp hasn't updated to the
  post-rebrand ticker other exchanges use, a naming trap for
  `resolve_instrument`, not a real gap); 1 via fiat conversion (`ZEC`
  only has `ZEC/USD` on Bitstamp, no `ZEC/EUR` — but `EUR/USD` is itself
  a liquid Bitstamp pair, so EUR→USD→ZEC is a free two-hop route, same
  logic Baz flagged for stablecoin conversion generally).
- **13 confirmed genuinely absent** (zero listing under any of Bitstamp's
  10 quote currencies — USD/EUR/GBP/USDT/USDC/BTC/RLUSD/EURC/EURCV/ETH,
  so no fiat/stablecoin conversion route exists either — not a
  EUR-quoting artifact, checked the same way as OKX's 8): `FIL`, `VET`,
  `EOS`, `THETA`, `RUNE`, `ENJ`, `DASH`, `KAVA`, `1INCH`, `BEAM`, `GALA`,
  `TIA`, `USUAL`.
- This is meaningfully **narrower coverage than OKX (87%)** — Bitstamp is
  the older, smaller exchange of the two.

**New candidates**: checked both EUR-quoted assets AND assets with only a
non-EUR quote (USD/USDT/USDC/GBP/BTC/ETH), since those are equally
reachable via Bitstamp's own EUR/USD and stablecoin pairs. The
non-EUR-only search turned up only 4 bases beyond the EUR-quoted set —
`ETH2` (a staking derivative), `SGD`/`XSGD` (Singapore-dollar/tokenized
fiat), `USDG` (a stablecoin) — none are directional-prediction
candidates, so the EUR-quoted search already covered everything real. Of
Bitstamp's 59 non-universe EUR-quoted assets (excluding fiat/stablecoin
quotes and the RNDR/RENDER alias above), most have **zero 24h trading
volume** on Bitstamp specifically (checked via live `ticker` calls, not
assumed — e.g. `EGLD/EUR`, `SEI/EUR`, `APE/EUR` all returned
`"volume":"0.00000"`, genuinely dead pairs, not a data error). Using the
same JTO-EUR ~8,139 EUR/24h floor as OKX, only **4 clear it**:

| Symbol | 24h EUR volume | Notes |
|---|---|---|
| CASHCAT | 258,950 | Very high volume for an unfamiliar name — worth a sanity check before trusting, possible thin/manipulated order book on a low-cap meme coin |
| ZORA | 37,473 | Zora Network |
| FET | 27,292 | Fetch.ai — **also an OKX candidate** (20,013 there), same asset found independently on two exchanges |
| HYPE | 9,656 | Hyperliquid — **also an OKX candidate** (829,297 there, far more liquid there) |

`VIRTUAL` (6,693 EUR/24h) is the closest miss, just under the floor.
Everything else in the 59 was either a dead pair (0 volume) or below
6,693.

**Practical implication for exchange selection, not just universe
coverage**: if Bitstamp is chosen as Kairos's actual execution venue, its
narrower listing (77% vs. OKX's 87%) and mostly-illiquid long tail matter
independently of its fee schedule — the fee comparison in
`docs/exchanges/README.md` doesn't capture this.

## Bybit EU (2026-09-13)

Checked Kairos's full 62-symbol crypto universe against Bybit EU's public
`GET /v5/market/instruments-info` (131 spot pairs total, quoted only in
EUR/USDC/PLN — one call covers everything, no per-symbol lookup needed).

- **41/62 (66%) are tradeable against EUR, directly or via free
  conversion** — 12 have a direct EUR pair; the other 29 are USDC-quoted
  only, but Bybit EU lists a genuine `USDC/EUR` spot pair (confirmed live,
  not assumed), so USDC is itself a free EUR-conversion hop. Applied
  `feedback_exchange_coverage_indirect_routes.md`'s rule from the start
  this time, not as a correction afterward like Bitstamp's ZEC case.
- **21 confirmed genuinely absent** (no pair under any of Bybit EU's 3
  quote currencies): `ETC`, `VET`, `MKR`, `GRT`, `AXS`, `EOS`, `XTZ`,
  `THETA`, `RUNE`, `SNX`, `ENJ`, `CHZ`, `ZEC`, `DASH`, `KAVA`, `1INCH`,
  `LDO`, `BEAM`, `JTO`, `TON`, `USUAL`.
- This is **narrower than both OKX (87%) and Bitstamp (79%)** — Bybit EU's
  131-pair spot listing is much smaller than either, consistent with it
  being the newest MiCA entity of the three and carved out of a much
  larger global listing that EU/EEA customers can't reach.

**New candidates**: checked all EUR/USDC-quoted bases outside the universe
against live 24h turnover, same ~8,139 EUR/24h floor used for OKX/Bitstamp.
16 cleared it; excluding `EURC` (a stablecoin, not a directional-prediction
candidate), **15 real candidates** — 5 overlap with OKX's own candidate
list (noted below), 10 are new-to-either-list:

| Symbol | 24h EUR volume | Notes |
|---|---|---|
| PUMP | 320,785 | pump.fun — **also an OKX candidate** (633,421 there, more liquid there) |
| KII | 263,664 | |
| GROVE | 76,614 | |
| APT | 52,422 | Aptos |
| AERO | 44,365 | Aerodrome Finance |
| LIT | 41,779 | Litentry |
| GOAT | 30,787 | |
| MNT | 29,829 | Mantle |
| VIRTUAL | 27,291 | Virtuals Protocol — **also an OKX candidate** (17,518 there) — more liquid here |
| TRUMP | 19,821 | political meme coin — **also an OKX candidate** (74,253 there, more liquid there) |
| TRIA | 19,748 | |
| SEI | 17,080 | |
| PENGU | 15,634 | Pudgy Penguins — **also an OKX candidate** (44,500 there, more liquid there) |
| XPL | 10,055 | **also an OKX candidate** (26,290 there, more liquid there) |
| SPX | 9,004 | |

None overlap with Bitstamp's 4 (`CASHCAT`/`ZORA`/`FET`/`HYPE`).

**Practical implication**: if execution-venue selection weighs listing
breadth, Bybit EU's spot market is the smallest of the three live-tested
crypto exchanges so far — its case (if any) would need to rest on cost or
API quality, not universe coverage.

## Kraken / Bitvavo — pending

No demo/sandbox account exists for either yet (see `docs/exchanges/README.md`
— both need a real funded account, no self-service sandbox). Deprioritized
relative to the exchanges with free demo access.
