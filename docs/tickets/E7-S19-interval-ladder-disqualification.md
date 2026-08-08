# E7-S19 — Interval-ladder resolution & disqualification

**Goal:** For a given signal, determine which interval (smallest-first) has enough forward price data to resolve its outcome, or disqualify it if none do.

**Context:**
- Read `docs/tickets/DESIGN_DOC_offline_signal_replay.md` §3.2 in full — the "smallest available interval, per (ticker, signal)" rule and the "disqualified means absent, no interpolation" rule are both hard requirements, not suggestions.
- Read the installed `phantom` package's `price_cache.get_price_data(ticker, start_date, end_date, interval, db_path)` (`.venv/lib/python3.13/site-packages/phantom/data/price_cache/src/price_cache/_cache.py`, search for `def get_price_data`) — the primitive this story calls per candidate interval. Confirm its exact signature/return type (`Optional[pd.DataFrame]`, tz-aware `DatetimeIndex`, `Open/High/Low/Close/Volume` columns) against current source.
- Read `strategy/kairos_papertrade.py`'s `_IntradayFallbackProvider.get_bars` (~line 281-330) for the existing precedent of trying multiple intervals in a ladder, falling back to the next on empty/failed fetch — mirror this pattern's shape, don't diverge from it without reason.

**Acceptance criteria:**
- [ ] `resolve_interval_for_signal(ticker: str, entry_datetime, interval_ladder: list[str], min_bars: int, db_path: str) -> str | None`: tries each interval in `interval_ladder`, IN THE ORDER GIVEN (caller is responsible for passing smallest-first; this function does not re-sort), calling `price_cache.get_price_data(ticker, start=entry_datetime, end=<a reasonable forward window>, interval=interval, db_path=db_path)` for each. Returns the FIRST interval whose returned DataFrame has at least `min_bars` rows. Returns `None` if every interval in the ladder yields fewer than `min_bars` rows (or `None`/empty).
- [ ] `min_bars` has a small sensible default (e.g. `2`) — closure resolution needs enough forward bars to potentially hit a stop/target or exhaust the resolution window, not a large lookback.
- [ ] Function does not raise on a `price_cache.get_price_data` call that returns `None` or an empty DataFrame for one interval — it moves to the next interval in the ladder.
- [ ] Unit tests mock `price_cache.get_price_data` (patch the function reference `kairos_signal_replay`'s module imports it under) to return: (a) sufficient data on the first/smallest interval — assert that interval is returned without trying others (assert the mock was called once); (b) empty on the smallest, sufficient on a later interval — assert fallback works and the correct interval is returned; (c) empty on every interval — assert `None` is returned.

**Definition of done:**
- [ ] `flake8`/`mypy` pass (scoped).
- [ ] Interval-ladder tests pass.
- [ ] Changes committed and `docs/todo.md` E7-S19 item checked off.
