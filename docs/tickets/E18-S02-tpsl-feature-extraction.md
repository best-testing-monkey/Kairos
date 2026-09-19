# E18-S02 — Price-history-derived feature extraction for TP/SL model

**Goal:** A pure function that extracts the ML model's input features for a given historical
signal, computed entirely from re-fetchable price history (no GPU, no re-running predictions) —
plus a unit test proving it never reads data beyond the signal's own `as_of` timestamp.

**Context — read this scoping note before writing anything:**
`DESIGN_DOC_ml_tpsl_optimization.md` §4's factor list includes distribution-derived stats
(`KairosDistribution.entropy()`/`kurt`/`skew` — `kairos_backtest.py:346`, `:286-341`). **Those are
deliberately OUT OF SCOPE for this story.** Historical signals in `papertrade_signals` do not
persist the full 100-sample prediction distribution that produced them (only scalar
`entry`/`stop`/`target`/`expected_value`/`base_win_rate` are stored) — reconstructing entropy/kurt
for a past signal would mean re-running the model, which defeats the "cheap, offline, no GPU" goal
this whole design exists for. v1's feature set is **price-history-only**: everything below is
reconstructable by re-fetching historical OHLCV bars for `(ticker, as_of)`, the same way
`kairos_signal_replay.py` already does. Dist-derived features are a documented future enhancement,
gated on `signals_cache`/`kairos_predcache` starting to persist distribution summary stats
going forward — not this story's job.

- New module: `strategy/kairos_tpsl_features.py`.
- Feature function signature: `extract_features(ticker: str, as_of: str, interval: str, entry: float,
  history: pd.DataFrame) -> dict[str, float]` — `history` is real OHLCV bars up to and including
  `as_of`, already fetched by the caller (mirror `kairos_signal_replay.py`'s
  `_ensure_configured_db()` + `price_cache.get_price_data()` pattern for how the caller obtains
  it — this function itself takes `history` as an argument and does no fetching, to keep it a pure,
  trivially-testable function).
- Features to compute (all price-history-only, per the recurring-factor list in
  `DESIGN_DOC_ml_tpsl_optimization.md` §3 minus the dist-derived ones):
  - ATR(14) — reuse the exact formula `ATRBracketStrategy` already uses
    (`strategy/kairos_volatility.py:170-245`, read its ATR calculation and copy the same formula,
    don't reinvent it).
  - Realized volatility — rolling stdev of returns over a fixed window (e.g. 20 bars).
  - Asset class — call `kairos_strategies.asset_class_for()` (`strategy/kairos_strategies.py:869`)
    on `[ticker]`. Do not write a second classifier — this repo's CLAUDE.md "Four classifiers"
    section is explicit that duplicating this logic is a foot-gun.
  - Interval/horizon — pass through the `interval` string as a categorical feature.
  - Trend/momentum — e.g. `(close[-1] - close[-N]) / close[-N]` for some fixed N (pick 10, document
    the choice).
  - Position within recent range — `(close[-1] - low_N) / (high_N - low_N)` over a fixed lookback.
- **No-lookahead test is the point of this story.** Assert the function's output is unchanged
  whether `history` is exactly truncated at `as_of` or has extra rows appended *after* `as_of` —
  i.e., feed it `history.loc[:as_of]` vs a longer `history` slice and require byte-identical
  output. This mirrors the pattern in `tests/unit/test_naive_no_lookahead.py` (read it for the
  existing convention this repo uses for exactly this class of test) — reuse that file's
  structure/style rather than inventing a new one.

**Acceptance criteria:**
- [ ] `extract_features()` implemented in `strategy/kairos_tpsl_features.py`, returns a flat
  `dict[str, float]` (categorical features like `asset_class`/`interval` encoded as strings in the
  dict — encoding to numeric for the model is E18-S03's job, not this one).
- [ ] ATR calculation matches `ATRBracketStrategy`'s own ATR value on the same input (assert
  equality in a test using a shared fixture DataFrame).
- [ ] Unit test: no-lookahead invariant (above) passes for at least 2 different `as_of` cutoffs.
- [ ] Unit test: function raises/handles cleanly (not a silent `NaN`) when `history` has fewer rows
  than the largest lookback window used internally — mirrors this repo's existing "insufficient
  history" guards (e.g. `GBMDirectionStrategy`'s `if len(history) < 120: return None` in
  `kairos_ml.py`, though this function returns/raises rather than returning `None` since it's not a
  `Signal`-producing strategy).
- [ ] `asset_class_for()` is called, not reimplemented.

**Definition of done:**
- [ ] `flake8`/`mypy` pass, scoped to `strategy/kairos_tpsl_features.py`.
- [ ] New tests pass (`tests/unit/test_tpsl_features.py`); full suite green.
- [ ] Changes committed and `docs/todo.md` E18-S02 item checked off.
