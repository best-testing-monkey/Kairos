---
name: run-stories
description: Execute the next unchecked stories from todo.md via cheap-model subagents, one per story, with gates and commit per story. Use when the user says /run-stories, "implement the stories", "have haiku work through the todo", or after /breakdown completes.
---

# /run-stories — implement todo.md via cheap subagents

Prereq: `docs/todo.md` + `docs/tickets/` exist (else run `/breakdown` first). Follow the `cheap-models-first` ladder — you are the orchestrator; you do not implement.

## Loop (repeat until todo done or a story blocks)

1. **Pick** the next unchecked story in `docs/todo.md` (respect order — it encodes dependencies). Independent stories in the same epic MAY run as parallel subagents if they touch disjoint files.

2. **Spawn** one subagent per story with an explicit model:
   - `model: "haiku"` — default for spec'd stories.
   - `model: "sonnet"` — UI stories, anything with judgment calls, or any story a Haiku agent already failed once.
   Context bundle (nothing more): the ticket file, `APPENDIX-A-standards.md`, the project skill's file map section, and the exact files listed in the ticket's Context.
   Required instructions to the agent: implement to the acceptance criteria; run the gates (`uv run ruff/flake8`, `mypy` if configured, `uv run pytest tests/ -q`); commit with a message referencing the story ID; report test output as evidence.

3. **Verify, don't trust**: check the agent's reported gate output; spot-check the diff (`git show --stat`). Only then mark `- [x]` in `docs/todo.md` (amend into the story commit or a small follow-up commit).

4. **On failure**: same story fails twice at a rung → escalate one rung (haiku→sonnet→inline). Never silently absorb the work inline without noting the escalation.

## After the last story

Run `/verify` on the integrated result (drive the real app — server up, key flows exercised, e.g. `scripts/qa_snapshot.py` in phantom_ledger). Then suggest `/code-review` on the branch. Report: stories completed, escalations, anything left unchecked and why.
