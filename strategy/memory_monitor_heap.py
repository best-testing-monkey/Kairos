"""In-process RSS watchdog, imported unconditionally by kairos_papertrade.py
(see the `import memory_monitor_heap` line near its top) as the last line of
defense against the same class of runaway-memory freeze that repeatedly took
the machine down in early August 2026 while root-causing kairos_predcache's
and kairos_strategies' overlapping prediction caches (see CLAUDE.md's
"Prewarm leak sources" section for that history). Importing this module has
the side effect of starting a daemon thread (`monitor_thread`, started at
import time below) that polls this process's own RSS every CHECK_INTERVAL
(0.5s) seconds via psutil, and the moment it crosses THRESHOLD_MB (6000MB):

  1. Dumps the top 15 tracemalloc allocation sites (filtered to frames under
     this repo's own path, excluding site-packages) by size -- tracemalloc
     is started at import time (`tracemalloc.start()` below) so every
     allocation from process start is attributed to its call site.
  2. Hard-exits the process (`os._exit(1)`) -- deliberately NOT a clean
     `sys.exit()`, since the whole point is to stop growing before the
     8GB-class cgroup/OOM kill that used to only get caught externally (by
     hand, or by a wrapping systemd-run --scope -p MemoryMax=... during
     live debugging) has a chance to happen, or worse, before the box starts
     swapping and freezes outright with no diagnostic dump at all.

This used to also suspend the main thread first (via
ctypes.pythonapi.PyThreadState_SetAsyncExc, injecting an async RuntimeError)
so the snapshot below wasn't racing further allocations. Removed
2026-08-13 after a live run deadlocked the entire process for 20+ minutes
with zero output: the async exception landed mid-way through
`Kronos.from_pretrained()`'s CUDA/safetensors calls in
kairos_strategies._materialize_model(), an upstream retry then re-entered
with the CUDA/allocator state left half-built by the aborted first attempt
(visible in the log as "Loading Kronos model from ..." printed twice with
no intervening "Switching Kronos model:" line -- only possible if the first
attempt was interrupted after `bt_predictor` was already nulled out), and
that deadlocked the main thread inside a C call that never yields the GIL.
Every other thread -- including this monitor's own timeout-guarded
tracemalloc calls -- needs the GIL too, so nothing could run, not even the
code that would print "ran out of time". The whole safety net went dark.
The main thread being a few hundred ms further along by the time the
snapshot runs is harmless; a hard deadlock during GPU model loading is not.

THRESHOLD_MB (6000) is deliberately well under a typical cgroup/container
cap (8GB was used during the live debugging that motivated this file) so
there's headroom for the dump itself and for whatever RSS the process is
using at the moment of the check, not just the threshold value.

analyze_heap()/find_variable_by_object()/find_referrer_info() are a second,
currently-disabled (commented out in memory_monitor()) diagnostic pass that
walks gc.get_objects() for large live objects and tries to name the
variable/dict-key/frame referencing each one -- much slower and more
invasive than the tracemalloc dump above (calls gc.collect() and iterates
every tracked object), kept here for a deeper manual dig if the tracemalloc
top-15 alone isn't enough to identify a future leak.
"""
import psutil
import os
import sys
import signal
import threading
import tracemalloc
import tempfile
import time
import gc
import inspect

THRESHOLD_MB = 6000
CHECK_INTERVAL = 0.5
GROWTH_STEP_MB = 1000  # take a lightweight tracemalloc diff every this much RSS growth
ANALYSIS_TIMEOUT_S = 180  # wall-clock budget for one tracemalloc analysis subprocess
                          # (see _run_in_subprocess -- this is enforced with SIGKILL, not
                          # a thread join, because a thread join can't actually preempt
                          # tracemalloc's C-level snapshot call; see its docstring for why
                          # that matters).

process = psutil.Process(os.getpid())
tracemalloc.start(1)  # nframe=1: minimum traceback depth (this is already tracemalloc's
                       # default -- made explicit since it's the main scope-reduction lever
                       # available here; every extra frame multiplies the size of every
                       # tracked allocation's traceback across the whole live table)

_last_growth_snapshot_path = None
_last_growth_snapshot_mb = 0.0


def _run_in_subprocess(fn, timeout=ANALYSIS_TIMEOUT_S, label="analysis"):
    """Run fn() in a forked child process, SIGKILLed if it outlives timeout.

    A `threading.Thread` + `.join(timeout)` (the first version of this
    function) cannot enforce a real wall-clock timeout here:
    `tracemalloc.take_snapshot()` holds the GIL for its entire duration on a
    heap this large, so the *watcher* thread's own "ran out of time" print
    can't get scheduled either -- it needs the GIL too, and never gets it
    back until the call finishes on its own. Observed live twice
    (2026-08-13): the process hung 20-30 minutes with zero further output
    despite a 180s thread-join timeout, saved only by the outer
    `systemd-run --scope -p MemoryMax=...` cgroup cap, not by this module.

    fork() sidesteps this: the child gets its own independent GIL in its
    own process (a COW copy of tracemalloc's tracked state at fork time),
    so it can churn on a huge snapshot without blocking the parent at all,
    and the parent can kill it with SIGKILL -- an OS-level, non-cooperative
    signal -- regardless of what C code the child is stuck in.

    fn() must do all of its own printing/file-writing and take no arguments
    (use a closure); nothing is returned across the fork boundary. Do not
    call this with a callable that touches CUDA/torch -- forking a process
    with an initialized CUDA context and then using CUDA in the child is
    unsafe. tracemalloc/gc/file I/O only.
    """
    pid = os.fork()
    if pid == 0:
        try:
            fn()
        except Exception as e:
            print(f"{label} failed in child: {e}", flush=True)
        finally:
            os._exit(0)

    deadline = time.time() + timeout
    while True:
        wpid, _status = os.waitpid(pid, os.WNOHANG)
        if wpid == pid:
            return True
        if time.time() >= deadline:
            print(f"{label} ran out of time ({timeout}s) -- killing analysis subprocess", flush=True)
            try:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
            except ProcessLookupError:
                pass
            return False
        time.sleep(0.2)


def _print_tracemalloc_top(snapshot, limit=15):
    top_stats = snapshot.statistics('lineno')
    count = 0
    for stat in top_stats:
        frames: tracemalloc.Traceback = stat.traceback
        for frame in frames:
            if not ("site-packages" in f"{frame.filename}") and frame.filename.startswith("/") and ("/Kairos/" in f"{frame.filename}"):
                print(f"{count + 1:2d}. {frame.filename}:{frame.lineno} size={stat.size / 1024 / 1024:.2f}MB count={stat.count}", flush=True)
                count += 1
                break
        if count >= limit:
            break


def _print_tracemalloc_growth(old_snapshot, new_snapshot, limit=15):
    diffs = new_snapshot.compare_to(old_snapshot, 'lineno')
    count = 0
    for stat in diffs:
        if stat.size_diff <= 0:
            continue
        frames: tracemalloc.Traceback = stat.traceback
        for frame in frames:
            if not ("site-packages" in f"{frame.filename}") and frame.filename.startswith("/") and ("/Kairos/" in f"{frame.filename}"):
                print(f"{count + 1:2d}. {frame.filename}:{frame.lineno} +{stat.size_diff / 1024 / 1024:.2f}MB (now {stat.size / 1024 / 1024:.2f}MB) count+={stat.count_diff}", flush=True)
                count += 1
                break
        if count >= limit:
            break
    if count == 0:
        print("(no single line accounts for the growth -- spread across many small allocations)", flush=True)


def _dump_top(limit=15):
    """Child-side (see _run_in_subprocess): take a snapshot and print its top-N."""
    snapshot = tracemalloc.take_snapshot()
    _print_tracemalloc_top(snapshot, limit=limit)


def _diff_and_dump(old_dump_path, new_dump_path, limit=15):
    """Child-side (see _run_in_subprocess): print growth since the previous
    checkpoint's dump (if any), then save this checkpoint's snapshot to
    new_dump_path so the *next* checkpoint's child can load it in turn.
    Snapshot objects can't cross the fork boundary back to the parent, so
    the parent only ever holds a plain file path, not a Snapshot -- this
    function does the load/compare/save entirely inside the child.
    """
    new_snapshot = tracemalloc.take_snapshot()
    if old_dump_path and os.path.exists(old_dump_path):
        old_snapshot = tracemalloc.Snapshot.load(old_dump_path)
        _print_tracemalloc_growth(old_snapshot, new_snapshot, limit=limit)
        os.remove(old_dump_path)
    new_snapshot.dump(new_dump_path)


def find_variable_by_object(obj, depth=3):
    """Traverse frames to find variable names that reference obj"""
    frame = sys._getframe(depth)
    candidates = []
    
    while frame:
        for var_name, var_obj in frame.f_locals.items():
            if var_obj is obj:
                candidates.append((var_name, frame.f_code.co_filename, frame.f_lineno))
            elif isinstance(var_obj, dict):
                for k, v in var_obj.items():
                    if v is obj:
                        candidates.append((f"{var_name}[{repr(k)}]", frame.f_code.co_filename, frame.f_lineno))
            elif isinstance(var_obj, (list, tuple)):
                for i, item in enumerate(var_obj):
                    if item is obj:
                        candidates.append((f"{var_name}[{i}]", frame.f_code.co_filename, frame.f_lineno))
        frame = frame.f_back
    
    return candidates

def find_referrer_info(obj):
    """Find the most direct referrer (frame variable or dict key)"""
    referrers = gc.get_referrers(obj)
    for ref in referrers:
        ref_type = type(ref).__name__
        if ref_type == 'frame':
            frame = ref
            for var_name, var_obj in frame.f_locals.items():
                if var_obj is obj:
                    return var_name, frame.f_code.co_filename, frame.f_lineno, frame.f_code.co_name
        elif ref_type == 'dict':
            for k, v in ref.items():
                if v is obj:
                    return repr(k), None, None, None
        elif ref_type == 'list':
            try:
                idx = ref.index(obj)
                return f"[{idx}]", None, None, None
            except ValueError:
                pass
    return None, None, None, None

def analyze_heap(deadline_s=ANALYSIS_TIMEOUT_S):
    """Quickly identify largest objects by syscall size, skip recursive traversal.

    gc.get_objects() on a long prewarm run can return millions of live
    objects; calling find_referrer_info() (itself a gc.get_referrers() scan)
    on each one is O(n) per object, so this used to hang indefinitely with
    no output (confirmed live by Baz running an earlier version of this
    function by hand). Bounded by a wall-clock deadline instead: stop
    scanning and report how far it got rather than never returning.
    """
    try:
        gc.collect()
    except:
        pass

    object_sizes = []
    deadline = time.time() + deadline_s
    scanned = 0
    all_objects = gc.get_objects()

    for obj in all_objects:
        scanned += 1
        if scanned % 2000 == 0 and time.time() > deadline:
            print(f"analyze_heap ran out of time ({deadline_s}s) -- scanned {scanned}/{len(all_objects)} objects", flush=True)
            break
        try:
            size = sys.getsizeof(obj)
            if size > 1024 * 100:
                type_name = type(obj).__name__
                var_name, filepath, lineno, func_name = find_referrer_info(obj)
                object_sizes.append((size, type_name, obj, var_name, filepath, lineno, func_name))
        except (TypeError, AttributeError, RuntimeError):
            pass

    object_sizes.sort(reverse=True)

    return object_sizes[:50]

def memory_monitor():
    """Polling loop run on `monitor_thread` (daemon, started at import time
    below). See this module's docstring for the threshold-crossing behavior;
    this function is just the sleep/check loop around it."""
    global _last_growth_snapshot_path, _last_growth_snapshot_mb
    while True:
        try:
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb >= THRESHOLD_MB:
                print(f"\n{'='*70}", flush=True)
                print(f"MEMORY THRESHOLD EXCEEDED: {memory_mb:.1f}MB", flush=True)
                print(f"{'='*70}\n", flush=True)

                print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                print(f"Current memory: {memory_mb:.1f}MB", flush=True)
                print(f"Memory percent: {process.memory_percent():.1f}%", flush=True)
                print(f"Virtual memory: {process.memory_info().vms / 1024 / 1024:.1f}MB\n", flush=True)

                print("TRACEMALLOC - Top 15 allocation sites:", flush=True)
                print("-" * 70, flush=True)
                _run_in_subprocess(_dump_top, label="tracemalloc dump")

                # print("\n" + "="*70)
                # print("HEAP ANALYSIS - Largest objects in memory:")
                # print("="*70 + "\n")
                
                # heap_objects = analyze_heap()
                
                # for i, (size, type_name, obj, var_name, filepath, lineno, func_name) in enumerate(heap_objects[:30], 1):
                #     size_mb = size / 1024 / 1024
                    
                #     var_display = var_name if var_name else "unknown"
                #     if filepath and lineno and func_name:
                #         location = f"{filepath}:{lineno} in {func_name}"
                #     else:
                #         location = "unknown location"
                    
                #     print(f"{i:2d}. {var_display:25s} {size_mb:8.2f}MB  (id: {id(obj)}, {location})")
                    
                #     if type_name == 'list':
                #         print(f"    list (Length: {len(obj)}, First few items: {str(obj[:3])[:70]})")
                #     elif type_name == 'dict':
                #         print(f"    dict (Keys: {len(obj)}, Sample keys: {str(list(obj.keys())[:3])[:70]})")
                #     elif type_name == 'ndarray':
                #         try:
                #             print(f"    ndarray (Shape: {obj.shape}, dtype: {obj.dtype})")
                #         except:
                #             print(f"    ndarray")
                #     elif type_name in ('str', 'bytes'):
                #         content_preview = str(obj)[:70]
                #         print(f"    {type_name} ({content_preview})")
                #     else:
                #         print(f"    {type_name}")
                    
                #     referrers = gc.get_referrers(obj)
                #     if referrers:
                #         print(f"    Referrers ({len(referrers)}):")
                #         for ref in referrers[:3]:
                #             ref_type = type(ref).__name__
                #             if ref_type == 'frame':
                #                 frame = ref
                #                 print(f"      - Frame: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
                #                 for vname, vobj in list(frame.f_locals.items())[:3]:
                #                     if vobj is obj:
                #                         print(f"        variable {vname}")
                #             elif ref_type == 'dict':
                #                 for k, v in list(ref.items())[:1]:
                #                     if v is obj:
                #                         print(f"      - Dict key: {repr(k)}")
                #             elif ref_type == 'list':
                #                 try:
                #                     idx = ref.index(obj)
                #                     print(f"      - List index: {idx}")
                #                 except ValueError:
                #                     pass
                #             else:
                #                 print(f"      - {ref_type}")
                #     print()

                print("="*70, flush=True)
                print("Memory monitor exiting.", flush=True)
                print("="*70 + "\n", flush=True)

                os._exit(1)

            elif memory_mb - _last_growth_snapshot_mb >= GROWTH_STEP_MB:
                # Lightweight periodic diagnostic, no thread suspension, no exit --
                # tracemalloc.compare_to() against the previous checkpoint shows what
                # actually grew in this ~1GB step, which is far more actionable than
                # the final threshold dump's single cumulative snapshot.
                print(f"\n--- Growth checkpoint: {memory_mb:.1f}MB (+{memory_mb - _last_growth_snapshot_mb:.0f}MB since last checkpoint) ---", flush=True)
                new_dump_path = os.path.join(tempfile.gettempdir(), f"kairos_tracemalloc_{os.getpid()}_{int(memory_mb)}.dump")
                old_dump_path = _last_growth_snapshot_path
                ok = _run_in_subprocess(lambda: _diff_and_dump(old_dump_path, new_dump_path), label="growth-checkpoint")
                if ok:
                    _last_growth_snapshot_path = new_dump_path
                elif old_dump_path:
                    # Killed mid-write: new_dump_path may be partial/missing, keep the old one.
                    try:
                        os.remove(new_dump_path)
                    except OSError:
                        pass
                # Reset tracemalloc's live tracking after every checkpoint so its
                # internal traceback table stays sized to one ~1GB window instead of
                # accumulating for the whole multi-hour run -- this is what actually
                # keeps take_snapshot() fast on checkpoint N+1; a longer timeout alone
                # wasn't enough because the table it has to walk kept growing every
                # checkpoint. Safe to clear here regardless of subprocess outcome: the
                # dump file (if written) is on disk, independent of tracemalloc's live
                # in-process state.
                tracemalloc.clear_traces()
                _last_growth_snapshot_mb = memory_mb  # advance even on a timed-out snapshot, so we don't retry every 0.5s

            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print(f"Monitor error: {e}", flush=True)
            raise e
            time.sleep(1)

monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
monitor_thread.start()

print(f"Memory monitor started (heap analysis enabled). Threshold: {THRESHOLD_MB}MB\n")


def _demo():
    """Self-check for _run_in_subprocess: a fast call finishes (True) and a
    call that outruns the deadline is SIGKILLed and reported (False), not
    hung on. Also exercises the growth-checkpoint dump/diff round trip."""
    marker = os.path.join(tempfile.gettempdir(), f"kairos_mmh_democheck_{os.getpid()}")
    assert _run_in_subprocess(lambda: open(marker, "w").close(), timeout=2, label="fast") is True
    assert os.path.exists(marker)
    os.remove(marker)

    start = time.time()
    assert _run_in_subprocess(lambda: time.sleep(5), timeout=0.3, label="slow (expected to time out)") is False
    assert time.time() - start < 4, "child wasn't actually killed -- it ran to completion instead of being SIGKILLed"

    dump1 = os.path.join(tempfile.gettempdir(), f"kairos_mmh_democheck_{os.getpid()}_1.dump")
    dump2 = os.path.join(tempfile.gettempdir(), f"kairos_mmh_democheck_{os.getpid()}_2.dump")
    assert _run_in_subprocess(lambda: _diff_and_dump(None, dump1), label="checkpoint 1") is True
    _ = [object() for _ in range(1000)]
    assert _run_in_subprocess(lambda: _diff_and_dump(dump1, dump2), label="checkpoint 2") is True
    assert not os.path.exists(dump1), "checkpoint 2 should have consumed and removed dump1"
    os.remove(dump2)

    print("memory_monitor_heap self-check OK")


if __name__ == "__main__":
    _demo()