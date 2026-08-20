# Hourly universe screen

Screen candidate assets for hourly (`1h`) bars — the first stage in the `--interval 1h`
pipeline. This playbook documents the `universe` stage for 1h; see
[daily-signals.md](daily-signals.md) for context on what the universe screen does.

## Prerequisites

- No GPU required — this is a data-quality check, not a model-inference stage.
- Network access for yfinance (this stage fetches bars for all candidate assets).

## Steps

```bash
uv run ./strategy/kairos_pipeline.py --stage universe --interval 1h
```

The stage will fetch 1h bars for all assets in `CANDIDATE_UNIVERSE` (defined in
`strategy/kairos_pipeline.py`) and compute liquidity metrics per asset. A row is
marked `PASS` if it passes all gates:

- **Sufficient bars:** at least `200 × 24 = 4800` hourly bars (20 days of continuous
  trading).
- **Dollar volume:** daily-equivalent threshold (e.g., $10M for crypto, $20M for
  equities) — the per-bar dollar volume is scaled up by 24 (BARS_PER_DAY for 1h).
- **Annualized volatility:** above a floor (e.g., 0.05 = 5% annual).
- **ATR:** at least 0.5% of price.
- **Interval probe:** for 1h specifically, a separate 5-day probe confirms the data
  source has recent data for the interval (failures here are rare but do happen when
  a data provider has data gaps; see "Caveats" below).

## What a successful run looks like

On a successful run, you'll see output like:

```
[...universe stage startup...]
  [     equity] SPY        PASS bars=5200 $vol=150000000.0 atr%=1.1 reason=None
  [     equity] QQQ        PASS bars=5200 $vol=98000000.0 atr%=1.5 reason=None
  [     equity] IWM        PASS bars=5200 $vol=45000000.0 atr%=0.8 reason=None
  [     crypto] BTC-USD    PASS bars=5200 $vol=125000000.0 atr%=2.3 reason=None
  [     crypto] ETH-USD    PASS bars=5200 $vol=62000000.0 atr%=1.8 reason=None
  [     crypto] XRP-USD    fail bars=5200 $vol=8000000.0 atr%=1.2 reason=low_dollar_volume (daily_equiv=192000000.0)
  [        FX ] EURUSD=X   PASS bars=5200 $vol=None atr%=0.6 reason=None
[...completed. N rows passed, M rows failed...]
```

- Each line is one asset; `PASS` / `fail` shows the gate result.
- `$vol` is the median daily-equivalent dollar volume (or the per-bar value for FX,
  which is exempt from this check).
- `daily_equiv=` appears in the reason only when dollar volume is the failure cause.
- All PASS rows are eligible for the next stages (`--stage correlation --interval 1h`,
  then `oracle`, then model training).

## Caveats

- **yfinance 729-day 1h history cap:** yfinance fetches at most 729 days of 1h bars.
  For intervals of 1d and higher, this is not a constraint (the code fetches 400 days).
  For 1h, yfinance's limit is firm — roughly 2 years of continuous history. If this
  becomes a bottleneck, see [ROADMAP.md](../../ROADMAP.md) Phase 5 (ccxt migration).
  This typically manifests as a `low_bars` failure for an old asset if you're running
  this very soon after adding it to `CANDIDATE_UNIVERSE`.

- **Daily-equivalent dollar-volume scaling:** The liquidity thresholds (e.g., $10M
  for crypto) are calibrated for daily bars. For 1h bars, the per-bar dollar volume
  is multiplied by 24 (BARS_PER_DAY for 1h) before comparing to the threshold. If
  you see a `low_dollar_volume` failure with a `daily_equiv=` note showing a value
  that passes when scaled, the scaling is working correctly — the asset genuinely
  trades less than the threshold on average, even accounting for intraday movement.

- **This is a pipeline entry point:** `--stage universe --interval 1h` must complete
  successfully before `--stage correlation --interval 1h` (the next stage, planned
  in Epic 11) can run. The `universe_screen` table persists PASS rows. If no assets
  pass this stage, correlation and all downstream stages will have nothing to work
  with.

- **Data-quality gaps:** yfinance is "marginal for hourly" per
  [ROADMAP.md](../../ROADMAP.md) — you may encounter occasional one-off data gaps
  (missing bars or exchanges temporarily offline). A single gap won't usually cause
  a failure (the bar count is usually still above 4800), but consistent gaps across
  multiple assets suggest a temporary data-source issue. Check
  [ROADMAP.md](../../ROADMAP.md) Phase 5 for the planned ccxt migration.

See also: [hourly-signals.md](hourly-signals.md) (generating signals from the
screened universe) and [weekly-strategy-discovery.md](weekly-strategy-discovery.md)
(oracle/model training, which depends on universe + signals for the 1h interval).
