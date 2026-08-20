# E17-S02 — Live-run the hourly digest manually; write the playbook; hand off the go/no-go on enabling the timer

**Goal:** Confirm `scripts/kairos_daily_signals.py --intervals 1h` produces a sane Telegram digest against a fully-built `1h` pipeline before the new systemd timer (E17-S01) is ever enabled for real unattended hourly runs.

**Requires GPU for model inference. Single invocation, not an open-ended loop — safe to run under light supervision.**

**Context:**
- Depends on E10–E16 all being done (the full `1h` pipeline needs to actually exist for a digest run to have anything real to report) and E17-S01 (the service/timer files exist, even if not yet enabled).
- Command: `uv run scripts/kairos_daily_signals.py --intervals 1h <same flags as systemd/kairos-hourly-signals.service's ExecStart>` — run it manually first, exactly as the timer would, before ever enabling the timer.
- Read `docs/tickets/DESIGN_DOC_multi_interval_1h.md` §2 (E8/E17 section) and `docs/playbooks/hourly-signals.md` in full.

**Acceptance criteria:**
- [ ] Run the exact command from `systemd/kairos-hourly-signals.service`'s `ExecStart` by hand (source `~/.config/kairos/kairos.env` first per CLAUDE.md's Telegram-notification note, or notifications will silently no-op).
- [ ] Confirm a Telegram message arrives (if signals were selected) or nothing arrives (if none were, given `--notify-empty` defaults off) — either is a valid outcome, confirm it matches what actually happened (don't just assume silence means success; check the generated report file too).
- [ ] Read the digest message content (prices, leverage, margin, EV-on-margin per the existing `format_allocation_row`/`build_success_message` logic) and confirm it reads sensibly for `1h`-cadence numbers (e.g. EV percentages should be smaller in magnitude than a `1d` digest's, per `hourly-signals.md`'s own caveat — a suspiciously large EV would suggest a unit/scaling bug somewhere upstream).
- [ ] Write `docs/playbooks/hourly-digest.md` (new file, mirror `docs/playbooks/hourly-signals.md`'s structure but focused on the Telegram-digest wrapper specifically, not the raw `kairos_signals.py` call): prerequisites, the exact `systemd`-equivalent command, what a real message looks like, and an explicit "before enabling the timer" checklist (this story's own acceptance criteria, condensed).
- [ ] Explicitly flag to Baz (in the session summary, not just buried in a file) that `systemctl --user enable/start kairos-hourly-signals.timer` is a deliberate operational decision still pending — this story verifies the digest WORKS, it does not turn on unattended hourly automation.

**Definition of done:**
- [ ] Manual run completed and reviewed.
- [ ] `docs/playbooks/hourly-digest.md` exists and is accurate.
- [ ] `docs/todo.md` E17-S02 item checked off.
- [ ] Baz has been told the timer is ready but not yet enabled, and is the one to decide when to flip it on.
