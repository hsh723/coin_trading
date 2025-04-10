from typing import Dict, List
from dataclasses import dataclass
import psutil

@dataclass
class HealthStatus:
    cpu_usage: float
    memory_usage: float
    disk_usage: Dict[str, float]
    error_rate: float
    response_time: float

class TaskHealthMonitor:
    def __init__(self, health_thresholds: Dict = None):
        self.thresholds = health_thresholds or {
            'max_cpu': 80.0,
            'max_memory': 85.0,
            'max_error_rate': 0.05
        }
        
    async def check_health(self, task_id: str) -> HealthStatus:
        """작업 상태 점검"""
        return HealthStatus(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=self._get_disk_usage(),
            error_rate=self._calculate_error_rate(task_id),
            response_time=self._measure_response_time()
        )
