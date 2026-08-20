# E16-S01 — Guard MTM/financing accrual to fire at most once per calendar day

**Goal:** `kairos_papertrade.py`'s day-loop unconditionally computes a full day's MTM snapshot + financing accrual on every iteration, regardless of `--interval`. For `1d` this is correct (one iteration per day). For `1h`, the SAME `day_start`/`day_end` window (always midnight-to-midnight, truncated from `effective_dt`) would be recomputed 24 times within one calendar day, and `compute_daily_financing_total` (which accrues a FULL day's financing rate each call) would silently over-charge financing ~24x. Add a guard so financing/MTM-snapshot accrual only actually fires once per calendar day no matter how fine `--interval`'s loop step is.

**Context:**
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §2 (E7/E16 section).
- `strategy/kairos_papertrade.py`, the day-loop (`for effective_dt, stats_rows, advice_rows in dated_rows:`, search for this exact loop line — it's a large function, look for `known_open_ids = set()` a few lines before the loop starts as an anchor).
- `day_start`/`day_end` computation (search `day_start = datetime(effective_dt.year, effective_dt.month, effective_dt.day)`, ~line 2102): `day_start = datetime(effective_dt.year, effective_dt.month, effective_dt.day)` / `day_end = day_start + timedelta(days=1)` — always a 24h window truncated to midnight, regardless of `--interval`.
- The financing/MTM block (search `compute_daily_financing_total(mtm_positions, day_bars, margin_config)`, ~line 2325, inside a comment block starting `# Daily MTM snapshot + financing accrual`): computes `financing_day`, subtracts it from `corrected_cash`, and writes one `kairos_mtm_daily`/`DailySnapshot` row via `_insert_mtm_daily_row`. This whole block runs unconditionally every loop iteration today.
- `strategy/kairos_mtm.py`'s `daily_financing()`/`compute_daily_financing_total()` divide by `360` (annualized rate), explicitly a **once-per-day** accrual convention (docstring: "financing is charged on positions open at bar close; the entry day counts, the exit day does not") — do NOT change the `/360` math itself (that's correct for a once-a-day call); the fix belongs entirely in the CALLER's cadence, not the financing formula.
- `kairos_mtm_daily` table already has a `date` primary/unique-ish column (via `_ensure_mtm_daily_table`/`_insert_mtm_daily_row`, search those names) — confirm whether it already has a uniqueness constraint on `date` (if so, a second same-day insert may already fail/replace rather than double-accrue at the DB level, but `corrected_cash -= financing_day` happens in Python BEFORE the DB write and would still double-subtract from the in-memory equity even if the DB row itself doesn't duplicate — the guard must live at the Python level, not rely on a DB constraint).

**Acceptance criteria:**
- [ ] Track the last calendar date financing/MTM accrual actually ran for, e.g. a `last_financing_date: date | None = None` variable initialized before the loop (near `corrected_cash = args.capital`, ~line 2082).
- [ ] Wrap the financing/MTM block (the `day_bars = _fetch_day_close_bars(...)` through the `snapshot = ...`/DB-write lines) in `if last_financing_date != day_start.date():` — only run it when the calendar day has actually advanced since the last time it ran. Set `last_financing_date = day_start.date()` after the block runs.
- [ ] For `--interval 1d` (today's only real usage): `day_start.date()` changes every single iteration (one iteration = one day), so this guard is a no-op — behavior byte-identical to before.
- [ ] For `--interval 1h`: the guard fires the financing/MTM block only on the FIRST iteration of each calendar day; the other ~23 hourly iterations that day skip it entirely (no financing subtracted, no snapshot row written) until the next calendar day begins.
- [ ] Unit test (find or create a test file exercising `kairos_papertrade.py`'s day-loop with a mocked/fixture-driven `dated_rows` — check `tests/unit/` for existing papertrade day-loop tests to extend rather than building a new harness from scratch): feed `dated_rows` with several same-day hourly `effective_dt` entries followed by a next-day entry; assert `compute_daily_financing_total`/the MTM snapshot write is called exactly once per DISTINCT calendar date, not once per `dated_rows` entry.
- [ ] Existing `--interval 1d` papertrade tests (whatever currently exists) still pass unmodified — this is the core regression bar.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped to `strategy/kairos_papertrade.py`).
- [ ] New + existing tests pass; full suite green.
- [ ] Changes committed and `docs/todo.md` E16-S01 item checked off.
