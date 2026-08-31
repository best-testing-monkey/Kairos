#!/usr/bin/env bash
# Overnight chain: (A) model throughput comparison on a pinned group set, then
# (B) sweep as much of the base corpus as the remaining budget allows.
#
# Deliberately conservative, because it runs unattended:
#   * --gpu-workers 1 everywhere. Base peaks at 5124 MiB on a 4-asset group
#     (scripts/benchmark_models.py) and the card has 5.8GB usable, so one
#     prewarm worker is the only safe value. The wave-packing scheduler would
#     help on light groups but has never run live; an OOM in a phase-1 worker
#     can deadlock the pool rather than fail cleanly, and a deadlock wastes the
#     whole night silently.
#   * No --pipeline. Same reason: unproven for base, and it raises peak
#     concurrency.
#   * Each phase-A leg is `timeout`-bounded so one hung leg cannot eat the
#     night and starve phase B, which is the part that produces corpus value.
#   * Phase A never passes --model for base: that would write stage='base' rows
#     with a non-NULL model_path, breaking the all-NULL base convention and
#     making every resume query miss.
set -u

REPO=/media/baz/MonkeyWorks/PycharmProjects/Kairos
cd "$REPO" || exit 1
LOGDIR="${1:?usage: overnight_run.sh <logdir>}"
mkdir -p "$LOGDIR"
# NOT named GROUPS: that is a bash built-in readonly array holding the user's
# group IDs. Assigning to it is silently ignored and "$GROUPS" then expands to
# the primary GID (1000), so the first run tried to open a file called "1000".
GROUP_FILE="$REPO/data/throughput_groups.txt"

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "=== tree state (must be clean; subprocesses import what is on disk) ==="
git -C "$REPO" status --short | grep -vE '^\?\? docs/handoff' && say "WARNING: uncommitted changes present" || say "tree clean"
git -C "$REPO" log --oneline -1

# ---------------------------------------------------------------- phase A
say "=== PHASE A: throughput comparison, 20 pinned groups x 3 models ==="
for leg in "base:" "small:--model small" "mini:--model mini"; do
  stage="${leg%%:*}"; flag="${leg#*:}"
  t0=$(date +%s)
  say "--- $stage starting ---"
  # shellcheck disable=SC2086
  timeout 5400 uv run python -u scripts/run_model_parallel.py 1.5 \
      --stage "$stage" $flag \
      --assets-file "$GROUP_FILE" \
      --workers 4 --gpu-workers 1 --chunk-size 20 --cache-max-gb 8 \
      >> "$LOGDIR/throughput_${stage}.log" 2>&1
  rc=$?
  say "THROUGHPUT_RESULT stage=$stage seconds=$(( $(date +%s) - t0 )) rc=$rc"
  # Reap orphaned pool children: ProcessPoolExecutor workers survive a killed
  # parent, and leftovers hold GPU + skew whatever runs next (cost 25 groups
  # and every timing measurement earlier today).
  for p in $(pgrep -f 'kairos_strategies\.py' 2>/dev/null); do
    kill -TERM "$p" 2>/dev/null && say "reaped leftover $p"
  done
  sleep 5
done

# ---------------------------------------------------------------- phase B
say "=== PHASE B: base corpus sweep, remaining budget ==="
t0=$(date +%s)
# The parallel driver, not run_base_priority.py: its phase-2 replay parallelism
# is model-independent and proven on 180 small + 180 mini groups, and at
# --gpu-workers 1 it loads exactly one model at a time -- the same VRAM profile
# as the old serial script, so it is strictly faster at no extra risk.
# --model is deliberately OMITTED so model_path stays NULL, matching every
# existing stage='base' row; passing it would break the resume query.
uv run python -u scripts/run_model_parallel.py 7 --stage base \
    --workers 4 --gpu-workers 1 --chunk-size 16 --cache-max-gb 8 \
    >> "$LOGDIR/base_sweep.log" 2>&1
say "BASE_SWEEP_DONE seconds=$(( $(date +%s) - t0 )) rc=$?"

say "=== ALL DONE ==="
