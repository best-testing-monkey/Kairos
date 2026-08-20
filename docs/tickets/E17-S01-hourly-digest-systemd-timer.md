# E17-S01 — New systemd timer for hourly signals

**Goal:** `scripts/kairos_daily_signals.py` already supports `--intervals 1h` end-to-end (its `--intervals` CLI flag defaults to `["1d"]` and is forwarded straight through to `kairos_signals.py --intervals`) and already has a `--notify-empty` flag that defaults to off (so it's silent-unless-actionable by default, exactly what `docs/playbooks/hourly-signals.md` asks for under "Automation opportunities"). This story is almost entirely new systemd config, not code.

**Context:**
- Read `systemd/README.md` for the install/enable convention, and `systemd/kairos-daily-signals-nc-top8.service`+`.timer` as the exact template to copy (same `ExecStart` shape, same `EnvironmentFile=-%h/.config/kairos/kairos.env` pattern, same `Restart=no` comment about GPU recovery reboots).
- `scripts/kairos_daily_signals.py`'s `main()` CLI flags (search `add_argument` in that file) — confirm which flags matter for an hourly no-crypto-or-whatever-rule digest; a reasonable starting point is the SAME `--signal-selection`/`--cluster_map`/`--max-leverage`/`--margin-utilization` combination already used by `kairos-daily-signals-nc-top8.service`, just with `--intervals 1h` added and pointed at an hourly cadence — confirm with Baz before hardcoding a specific selection rule if unsure; the point of this story is the timer plumbing, not picking a new trading rule.
- `docs/playbooks/hourly-signals.md`'s own note: "Run this a few minutes past the top of the hour — `fetch_data_raw` rounds down to the last closed bar, so running too early just repeats the previous hour's bar." — the `OnCalendar` schedule must respect this (don't fire exactly on the hour).

**Acceptance criteria:**
- [ ] New `systemd/kairos-hourly-signals.service`: same shape as `kairos-daily-signals-nc-top8.service`, with `ExecStart` pointing at `scripts/kairos_daily_signals.py --intervals 1h ...` (carry over the same `--signal-selection`/`--cluster_map`/`--max-leverage`/`--margin-utilization` flags as the existing no-crypto daily service, unless told otherwise — keep it a straightforward copy-with-`--intervals-1h`, don't invent a new selection rule).
- [ ] New `systemd/kairos-hourly-signals.timer`: `OnCalendar=*-*-* *:05:00 UTC` (5 minutes past every hour, matching the "run a few minutes past the top of the hour" guidance) with `Persistent=true`.
- [ ] Update `systemd/README.md`'s install steps to mention the new unit pair (add it to the `cp`/`enable`/`start` command list alongside the existing three).
- [ ] Do NOT enable/start the timer as part of this story (no `systemctl --user enable/start` commands run) — that's an operational decision for Baz to make explicitly, not something this ticket should do unattended.

**Definition of done:**
- [ ] `systemd/kairos-hourly-signals.service` and `.timer` created, syntactically valid (check with `systemd-analyze verify systemd/kairos-hourly-signals.service` if `systemd-analyze` is available; otherwise visually diff against the working `kairos-daily-signals-nc-top8.service`/`.timer` pair).
- [ ] `systemd/README.md` updated.
- [ ] Changes committed and `docs/todo.md` E17-S01 item checked off.
