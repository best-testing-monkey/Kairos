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

## Bitstamp — pending

Not yet checked; do this once Bitstamp is connected (Baz's stated next
signup target).

## Bybit EU — pending

Not yet checked; Bybit EU is also crypto, verification still pending as
of 2026-09-06.

## Kraken / Bitvavo — pending

No demo/sandbox account exists for either yet (see `docs/exchanges/README.md`
— both need a real funded account, no self-service sandbox). Deprioritized
relative to the exchanges with free demo access.
