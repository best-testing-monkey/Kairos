# E1-S02 — Build margin config loader and symbol classifier

**Goal:** Create `strategy/kairos_margin.py` with `MarginConfig`, `MarginClass`, a YAML loader, and a pure `classify_symbol` function that replaces `_CFD_TICKER_RE` for margin purposes.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.1.
- Read `config/margin_ibkr.yaml` (output of E1-S01).
- Read `strategy/allocation.py` for dataclass style conventions (`AllocationConfig`).
- Read `strategy/kairos_papertrade.py` `map_instrument_type()` and `_CFD_TICKER_RE` to understand what is being replaced for margin only.
- Create/modify `strategy/kairos_margin.py`.
- Create/modify `tests/unit/test_kairos_margin.py`.

**Acceptance criteria:**
- [ ] `MarginConfig` dataclass loads `config/margin_ibkr.yaml` via a public function (e.g. `load_margin_config(path) -> MarginConfig`).
- [ ] `MarginClass` dataclass exposes: `name`, `initial_margin_pct`, `maintenance_margin_pct`, `financing_spread_pct`, plus `enabled` where relevant.
- [ ] `classify_symbol(symbol: str, cfg: MarginConfig) -> MarginClass` matches in this order:
  1. explicit `symbols` membership, in config class order;
  2. first matching `match` regex, in config class order;
  3. the class with `match: null`.
- [ ] `EURUSD=X`, `USDJPY=X`, etc. → `fx_major` with `initial_margin_pct == 3.33`.
- [ ] `GC=F`, `^GSPC`, `SPY`, `QQQ` → `index_gold_major` with `initial_margin_pct == 5.0`.
- [ ] `CL=F` → `commodity_other` with `initial_margin_pct == 10.0`.
- [ ] `BTC-USD` with `crypto_cfd.enabled=false` → `crypto_spot` with `initial_margin_pct == 100.0`.
- [ ] `AAPL` → `equity_cfd` with `initial_margin_pct == 20.0`.
- [ ] Per-symbol `overrides` entry wins over both explicit list and regex.
- [ ] Module imports no `phantom`, no GPU, no network libraries.
- [ ] `tests/unit/test_kairos_margin.py` covers all classification cases above and passes with `uv run --with pytest python -m pytest tests/unit/test_kairos_margin.py -q`.
- [ ] `map_instrument_type()` in `kairos_papertrade.py` is left untouched for phantom order placement.

**Definition of done:**
- [ ] `flake8` and `mypy` pass on new and touched files.
- [ ] All new unit tests pass.
- [ ] Changes committed and `docs/todo.md` E1-S02 item checked off.
