# Appendix A — Implementation Standards

Applies to every story in this breakdown. Do not inline these rules; reference this file.

## Project tooling

- **Package manager / runner:** `uv`. Always use `uv run <cmd>`; never bare `python`/`python3`.
- **One-off test deps:** `uv run --with pytest python -m pytest ...` if pytest is not already synced.

## Code style

- Max line length: 120 characters (`tool.flake8.max-line-length = 120` in `pyproject.toml`).
- Type hints: Python 3.11 syntax; prefer `str | None` over `Optional[str]`.
- Imports: group stdlib, third-party, then local; use absolute imports inside `kairos/`.
- Docstrings: every public module/function gets one; tag new modules with the `KAI-N` ticket style used elsewhere only if a matching ticket exists, otherwise omit.
- Naming: follow the existing repo (snake_case, descriptive).

## Error handling

- Raise typed exceptions from `kairos.errors` (`KairosError` hierarchy) for Kairos-specific failures.
- Generic `ValueError`/`RuntimeError` are acceptable only for truly internal/impossible states.

## Testing conventions

- **Pure modules** (`kairos_margin.py`, `kairos_mtm.py`, `kairos_signal_replay.py`) must be unit-testable with **no GPU, no network, and no `phantom` install**.
- Tests live in `tests/unit/`. Use `pytest`.
- To import strategy modules, start each test file with:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "strategy"))
  ```
- Use synthetic fixtures; mock external dependencies (`phantom`, price_cache, Telegram) when testing engine wiring.
- Run tests with `uv run --with pytest python -m pytest tests/unit/<file> -q`.

## Quality gates (run before every commit)

Scope `flake8`/`mypy` to the specific files your story touches — **do not** run them against
the whole repo (`kairos/`, `tests/`) or the whole `strategy/` tree. This codebase has large,
pre-existing, unrelated violations outside whatever module your story modifies; fixing those is
out of scope and will burn your whole budget on someone else's debt. Example, for a story that
only touches `strategy/kairos_signal_replay.py`:

```bash
uv run --with flake8 python -m flake8 strategy/kairos_signal_replay.py
uv run --with mypy python -m mypy strategy/kairos_signal_replay.py
uv run --with pytest python -m pytest tests/unit/test_kairos_signal_replay.py -q
```

Only your own new/changed lines need to be flake8/mypy-clean — if the tool reports a violation
on a line you didn't touch, that's pre-existing debt, not yours to fix (verify via `git diff`
or `git blame` before assuming otherwise). If a gate fails on YOUR lines, fix it before
committing. Do not widen thresholds or suppress warnings unless the story explicitly calls for
it.

## Git / commit rules

- Branch from `main` per story; small, single-purpose commits.
- Do **not** add a `Co-Authored-By` trailer.
- In the same commit that completes a story, mark its `docs/todo.md` item `[x]`.
- Do not commit generated outputs (`results/`, `output/`, `data/`, model weights, `.env`).

## Module-specific notes

- `strategy/` deliberately has **no `__init__.py`**. Import strategy modules by adding `strategy/` to `sys.path`; do not try to make it a package.
- `phantom` cash/equity numbers are **not** source of truth for margin math. All MTM and margin math is computed Kairos-side from position rows + price bars; phantom is used for order fill/SL/TP mechanics only.
- When touching `AllocationConfig`, keep defaults unchanged so existing callers are unaffected.
