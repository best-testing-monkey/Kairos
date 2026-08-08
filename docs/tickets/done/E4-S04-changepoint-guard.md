# E4-S04: Changepoint Guard

## Goal
Implement stateful Bayesian online changepoint detector that vetoes signals during regime breaks.

## Context
- Read: strategy/DESIGN_DOC_awesome_quant_gaps.md, section §4.4
- Files to modify/create: strategy/kairos_econometric.py, tests/unit/test_econometric.py
- Relevant existing code: None yet for changepoint detection
- Standards: see docs/tickets/APPENDIX-A-standards.md (must expose reset() method)

## Acceptance criteria
- ChangepointGuardStrategy filter wrapper (stateful, must expose reset() per APPENDIX-A standards)
- Bayesian online changepoint detection (Adams & MacKay algorithm)
- Hazard rate 1/60 (expect regime change ~every 60 days)
- Normal-Inverse-Gamma conjugate prior
- Vetoes all signals for cooloff_days (default 3) when P(run length < 5) > 0.5 (fresh regime break)
- Cooloff countdown verified, vetoes persist for exactly cooloff_days, verified in `test_changepoint_cooloff_duration`
- On synthetic mean-shift series, detects break within 3 days, verified in `test_changepoint_mean_shift_detection`
- <5% false-positive rate on white noise, verified in `test_changepoint_white_noise_false_positive_rate`
- reset() method clears state correctly per APPENDIX-A standards, verified in `test_changepoint_reset_clears_state`

## Definition of done
- Unit tests in `tests/unit/test_econometric.py` pass via `uv run --with pytest python -m pytest tests/unit/test_econometric.py -q`
- Standards in docs/tickets/APPENDIX-A-standards.md followed
- Changes committed per APPENDIX-A commit convention
- Story checked off in docs/todo.md
