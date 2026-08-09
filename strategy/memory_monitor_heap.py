import psutil
import os
import sys
import threading
import tracemalloc
import time
import gc
import inspect
import ctypes

THRESHOLD_MB = 4000
CHECK_INTERVAL = 0.5

process = psutil.Process(os.getpid())
main_thread_id = threading.current_thread().ident
tracemalloc.start()

def get_object_size(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    
    try:
        size = sys.getsizeof(obj)
    except (TypeError, AttributeError):
        return 0
    
    try:
        if isinstance(obj, dict):
            try:
                size += sum(get_object_size(v, seen) for v in list(obj.values()))
                size += sum(get_object_size(k, seen) for k in list(obj.keys()))
            except RuntimeError:
                pass
        elif isinstance(obj, (list, tuple)):
            try:
                size += sum(get_object_size(i, seen) for i in list(obj))
            except (RuntimeError, ValueError):
                pass
        elif hasattr(obj, '__dict__'):
            size += get_object_size(obj.__dict__, seen)
    except (TypeError, AttributeError, RecursionError):
        pass
    
    return size

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

def suspend_main_thread(main_thread_id):
    """Pause main thread by raising exception in it"""
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(main_thread_id),
        ctypes.py_object(RuntimeError)
    )
    if res == 0:
        pass
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(main_thread_id), None)

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

def analyze_heap():
    """Walk garbage collector heap and identify largest objects with variable names"""
    try:
        gc.collect()
    except:
        pass
    
    objects = gc.get_objects()
    object_sizes = []
    
    for obj in objects:
        try:
            size = get_object_size(obj)
            if size > 1024 * 100:
                type_name = type(obj).__name__
                var_name, filepath, lineno, func_name = find_referrer_info(obj)
                object_sizes.append((size, type_name, obj, var_name, filepath, lineno, func_name))
        except (TypeError, AttributeError, RecursionError, RuntimeError):
            pass
    
    object_sizes.sort(reverse=True)
    
    return object_sizes[:50]

def memory_monitor():
    while True:
        try:
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb >= THRESHOLD_MB:
                print(f"\n{'='*70}", flush=True)
                print(f"MEMORY THRESHOLD EXCEEDED: {memory_mb:.1f}MB", flush=True)
                print(f"{'='*70}\n", flush=True)
                suspend_main_thread(main_thread_id)
                time.sleep(0.5)                
                
                
                print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
                print(f"Current memory: {memory_mb:.1f}MB", flush=True)
                print(f"Memory percent: {process.memory_percent():.1f}%", flush=True)
                print(f"Virtual memory: {process.memory_info().vms / 1024 / 1024:.1f}MB\n", flush=True)
                
                print("TRACEMALLOC - Top 15 allocation sites:")
                print("-" * 70)
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                for i, stat in enumerate(top_stats[:15], 1):
                    print(f"{i:2d}. {stat}")
                
                print("\n" + "="*70)
                print("HEAP ANALYSIS - Largest objects in memory:")
                print("="*70 + "\n")
                
                heap_objects = analyze_heap()
                
                for i, (size, type_name, obj, var_name, filepath, lineno, func_name) in enumerate(heap_objects[:30], 1):
                    size_mb = size / 1024 / 1024
                    
                    var_display = var_name if var_name else "unknown"
                    if filepath and lineno and func_name:
                        location = f"{filepath}:{lineno} in {func_name}"
                    else:
                        location = "unknown location"
                    
                    print(f"{i:2d}. {var_display:25s} {size_mb:8.2f}MB  (id: {id(obj)}, {location})")
                    
                    if type_name == 'list':
                        print(f"    list (Length: {len(obj)}, First few items: {str(obj[:3])[:70]})")
                    elif type_name == 'dict':
                        print(f"    dict (Keys: {len(obj)}, Sample keys: {str(list(obj.keys())[:3])[:70]})")
                    elif type_name == 'ndarray':
                        try:
                            print(f"    ndarray (Shape: {obj.shape}, dtype: {obj.dtype})")
                        except:
                            print(f"    ndarray")
                    elif type_name in ('str', 'bytes'):
                        content_preview = str(obj)[:70]
                        print(f"    {type_name} ({content_preview})")
                    else:
                        print(f"    {type_name}")
                    
                    referrers = gc.get_referrers(obj)
                    if referrers:
                        print(f"    Referrers ({len(referrers)}):")
                        for ref in referrers[:3]:
                            ref_type = type(ref).__name__
                            if ref_type == 'frame':
                                frame = ref
                                print(f"      - Frame: {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}")
                                for vname, vobj in list(frame.f_locals.items())[:3]:
                                    if vobj is obj:
                                        print(f"        variable {vname}")
                            elif ref_type == 'dict':
                                for k, v in list(ref.items())[:1]:
                                    if v is obj:
                                        print(f"      - Dict key: {repr(k)}")
                            elif ref_type == 'list':
                                try:
                                    idx = ref.index(obj)
                                    print(f"      - List index: {idx}")
                                except ValueError:
                                    pass
                            else:
                                print(f"      - {ref_type}")
                    print()
                
                print("="*70)
                print("Memory monitor exiting.")
                print("="*70 + "\n")
                
                os._exit(1)
            
            time.sleep(CHECK_INTERVAL)
        except Exception as e:
            print(f"Monitor error: {e}", flush=True)
            raise e
            time.sleep(1)

monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
monitor_thread.start()

print(f"Memory monitor started (heap analysis enabled). Threshold: {THRESHOLD_MB}MB\n")