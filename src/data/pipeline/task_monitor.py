from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class TaskMetrics:
    execution_time: float
    memory_usage: float
    cpu_usage: float
    error_count: int

class TaskMonitor:
    def __init__(self):
        self.task_stats = {}
        self.alerts = []
        
    async def monitor_task(self, task_id: str) -> TaskMetrics:
        """작업 모니터링"""
        start_time = time.time()
        metrics = TaskMetrics(
            execution_time=0.0,
            memory_usage=self._get_memory_usage(),
            cpu_usage=self._get_cpu_usage(),
            error_count=0
        )
        
        try:
            await self._collect_metrics(task_id)
            metrics.execution_time = time.time() - start_time
            self.task_stats[task_id] = metrics
        except Exception as e:
            await self._handle_monitoring_error(e, task_id)
            
        return metrics
