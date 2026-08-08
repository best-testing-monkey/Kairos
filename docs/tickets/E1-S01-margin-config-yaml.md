# E1-S01 — Create default IBKR retail margin config file

**Goal:** Add `config/margin_ibkr.yaml` containing the default ESMA/IBKR retail margin rules used by the new margin model.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §4.1 and §3 for the exact default values.
- Output file: `config/margin_ibkr.yaml`.
- No code files are modified in this story.

**Acceptance criteria:**
- [ ] `config/margin_ibkr.yaml` exists and is valid YAML.
- [ ] Top-level keys: `base_currency`, `benchmark_annual_pct`, `negative_balance_protection`, `closeout_fraction`, `classes`, `overrides`, `short_borrow_annual_pct`.
- [ ] `base_currency: EUR`.
- [ ] `benchmark_annual_pct: 3.15`.
- [ ] `negative_balance_protection: true`.
- [ ] `closeout_fraction: 0.5`.
- [ ] `classes` contains exactly these keys in order: `fx_major`, `fx_minor`, `index_gold_major`, `commodity_other`, `crypto_cfd`, `crypto_spot`, `equity_cfd`.
- [ ] `crypto_cfd.enabled: false` by default.
- [ ] `crypto_spot` has `initial_margin_pct: 100.0`, `maintenance_margin_pct: 0.0`, `financing_spread_pct: 0.0`.
- [ ] `equity_cfd` is the default fallback (`match: null`).
- [ ] `overrides` is present as an empty map ready for per-symbol house margins.
- [ ] `short_borrow_annual_pct.default: 1.0` with empty `overrides`.
- [ ] A shell check or one-line Python script loads the file with `yaml.safe_load` without error.

**Definition of done:**
- [ ] File committed.
- [ ] YAML loads successfully.
- [ ] `docs/todo.md` E1-S01 item checked off in the same commit.
