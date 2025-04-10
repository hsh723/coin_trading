from typing import Dict, List
from dataclasses import dataclass
import asyncio
import psutil
import time

@dataclass
class TaskMetrics:
    task_id: str
    runtime: float
    memory_usage: float
    cpu_usage: float
    io_stats: Dict[str, float]
    error_count: int

class TaskMetricsCollector:
    def __init__(self, collection_interval: int = 5):
        self.collection_interval = collection_interval
        self.metrics_buffer = {}
        
    async def collect_metrics(self, task_id: str) -> TaskMetrics:
        """작업 메트릭스 수집"""
        start_time = time.time()
        process = psutil.Process()
        
        metrics = TaskMetrics(
            task_id=task_id,
            runtime=time.time() - start_time,
            memory_usage=process.memory_info().rss / 1024 / 1024,  # MB
            cpu_usage=process.cpu_percent(),
            io_stats=self._collect_io_stats(),
            error_count=self.metrics_buffer.get(task_id, {}).get('error_count', 0)
        )
        
        self.metrics_buffer[task_id] = metrics
        return metrics
        
    def _collect_io_stats(self) -> Dict[str, float]:
        """IO 통계 수집"""
        io_counters = psutil.disk_io_counters()
        return {
            'read_bytes': io_counters.read_bytes,
            'write_bytes': io_counters.write_bytes,
            'read_count': io_counters.read_count,
            'write_count': io_counters.write_count
        }
