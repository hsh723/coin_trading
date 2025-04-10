from typing import Dict, List
import cProfile
import pstats
from dataclasses import dataclass

@dataclass
class ProfileResult:
    task_id: str
    execution_time: float
    function_stats: List[Dict]
    memory_usage: Dict
    bottlenecks: List[str]

class TaskProfiler:
    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stats_history = {}
        
    async def profile_task(self, task_id: str, task_func: callable, *args, **kwargs) -> ProfileResult:
        """작업 프로파일링"""
        self.profiler.enable()
        result = await task_func(*args, **kwargs)
        self.profiler.disable()
        
        stats = pstats.Stats(self.profiler)
        return self._create_profile_result(task_id, stats)
