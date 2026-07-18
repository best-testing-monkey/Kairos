---
name: breakdown
description: Turn a design/architecture document into docs/tickets/ story files and an ordered todo.md, sized for cheap-model implementation. Use when the user says /breakdown, "create stories/tickets from this spec", "turn this design doc into a todo", or starts a new feature from a written design.
---

# /breakdown — design doc → tickets + todo.md

Input: a design/architecture doc (argument, or find it: `agent-docs/`, `docs/`, `*architecture*.md`, `DESIGN_DOC*.md`). Output: story files + ordered checklist. **The consumer of these stories is a Haiku subagent with limited context — write for that reader.**

## Steps

1. **Read the design doc fully.** Identify epics (subsystems/feature areas) and slice each into stories completable in one focused subagent run (~1-3 files touched, one testable behavior).

2. **Write one file per story** at `docs/tickets/E<n>-S<nn>-<slug>.md`:
   - **Goal** — one sentence.
   - **Context** — exact file paths to read/modify; relevant existing classes/functions by name (pull from the project skill's file map, don't make the Haiku agent explore).
   - **Acceptance criteria** — concrete, checkable bullets. For UI stories, include the exact URL/element/expected text to check (past Haiku UI stories with fuzzy criteria needed re-spawns).
   - **Definition of done** — gates pass, committed, todo item checked off.

3. **Write an implementation-standards appendix** once at `docs/tickets/APPENDIX-A-standards.md` (code style, error handling, test conventions, commit rules from CLAUDE.md / slice-workflow skill). Every story references it; never inline it per story.

4. **Write `docs/todo.md`**: checklist ordered by the dependency graph (a story appears only after everything it imports/depends on). Group by epic. Format: `- [ ] E1-S01 <title> (docs/tickets/E1-S01-*.md)`.

5. **Self-check before finishing** (past failure: first pass shipped headers without story files, then lacked acceptance criteria):
   - every todo entry has an existing ticket file
   - every ticket has acceptance criteria a Haiku agent can verify mechanically
   - no story requires reading files not listed in its Context

Then tell the user the epic/story count and suggest `/run-stories` to execute.
