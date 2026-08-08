# E5-S16 — Smoke test, docs update, and ticket closure

**Goal:** Run an integration smoke test, update README with the new papertrade flags, and mark `docs/papertrade_tickets/02-portfolio-exposure-cap.md` as subsumed.

**Context:**
- Read `docs/tickets/DESIGN_DOC_mtm_margin_leverage.md` §6.3 and §8.
- Read `README.md` papertrade section.
- Read `docs/papertrade_tickets/02-portfolio-exposure-cap.md`.
- Read `strategy/kairos_papertrade.py` CLI flags (output of E4-S08) and `kairos_mtm_daily` table.
- Modify `README.md` and `docs/papertrade_tickets/02-portfolio-exposure-cap.md`.

**Acceptance criteria:**
- [ ] A 2-week (`--months-back 0.5`) smoke test runs with `--max-leverage 2.0 --margin-utilization 0.8` on a small asset set without crashing.
- [ ] After the smoke test, `kairos_mtm_daily` contains rows and the JSON report contains the MTM metric block.
- [ ] `mtm_max_drawdown_pct >= pct_max_drawdown` on the smoke run.
- [ ] `README.md` papertrade section documents the new flags (`--margin-config`, `--max-leverage`, `--margin-utilization`) with one sentence each and example command.
- [ ] `docs/papertrade_tickets/02-portfolio-exposure-cap.md` is updated at the top with a note: "Subsumed by the margin model in docs/tickets/DESIGN_DOC_mtm_margin_leverage.md — admission check + utilization cap provides the exposure cap."
- [ ] No changes to `ROADMAP.md` or `strategy/PIPELINE.md`.

**Definition of done:**
- [ ] Smoke test completed successfully (or documented blocker if data/phantom unavailable).
- [ ] Docs committed.
- [ ] `docs/todo.md` E5-S16 item checked off in the same commit.
