import cProfile
import pstats
import memory_profiler
import time
from functools import wraps
from typing import Callable, Any

class Profiler:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.profiler = cProfile.Profile()
        
    def profile_time(self) -> Callable:
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                if not self.enabled:
                    return func(*args, **kwargs)
                    
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                
                print(f"{func.__name__} took {elapsed:.2f} seconds")
                return result
            return wrapper
        return decorator

    @staticmethod
    def profile_memory(func: Callable) -> Callable:
        @memory_profiler.profile
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)
        return wrapper

    def start_profiling(self) -> None:
        self.profiler.enable()

    def stop_profiling(self) -> None:
        self.profiler.disable()
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        stats.print_stats()
