---
name: slice-workflow
description: Baz's standard spec-driven development loop used across all PycharmProjects repos (Kairos, phantom_ledger, insider_trading_analysis, ...). Use whenever implementing a feature slice/story/ticket, working from a spec or todo.md, or before committing in any of these projects. Encodes uv-run discipline, quality gates, git/PR rules, and token-saving habits.
---

# Slice workflow (all PycharmProjects repos)

Standard loop: **read spec → pick next todo item → branch → implement → gates → commit → check off todo → (draft PR)**.

## Before starting
1. Read the project's spec doc first if one exists (check `agent-docs/`, `docs/`, `*architecture*.md`, `DESIGN_DOC*.md`), then `./todo.md` or `docs/todo.md` for the next unchecked item. Ticket details often live in `docs/tickets/` or `docs/todo/details/`.
2. Check for a project skill (`.claude/skills/*/SKILL.md`) — Kairos has `kairos-dev`, phantom_ledger has `phantom-dev`, insider_trading_analysis has `itp-dev`. They contain file maps: **use the map instead of re-reading core modules.**

## Python execution discipline
- These are all `uv` projects. **Always `uv run <cmd>`** — never bare `python`/`python3` (bare `python` is not on PATH here; bare `python3` misses the venv → ModuleNotFoundError).
- One-off deps: `uv run --with <pkg> python ...`.
- Tests: `uv run pytest tests/ -q --tb=short` for iteration; `-v` only when diagnosing.

## Quality gates (run BEFORE every commit)
```bash
uv run ruff check .            # or: uv run flake8 --max-line-length=120 (ITP)
uv run mypy --ignore-missing-imports .   # if mypy configured
uv run pytest tests/ -q
```
Respect pre-commit hooks if `.pre-commit-config.yaml` exists. Don't commit with failing gates.

## Git / PR rules
- Branch from `main` per slice; small, single-purpose commits.
- **Never add a Co-Authored-By trailer** (user preference, explicit in ITP sessions).
- Open PRs as **draft**.
- After each completed slice: mark the item `[x]` in todo.md in the same commit.

## Token-saving habits (evidence: 47% of past tool calls were re-exploration)
- Use native **Grep/Glob/Read tools**, not `bash grep`/`find` (past sessions: 747 shell greps, 0 native — native is cheaper and paginated).
- Read only the line ranges you need from large files; grep `def |class ` for a signature map instead of full reads.
- When spawning subagents to implement stories, hand them a fixed context bundle: the story file, implementation-standards appendix, the project skill's file map, and only the directly-touched source files.
- Long-running servers: start once in background with output to a log file, `curl` to verify — don't restart per check.
