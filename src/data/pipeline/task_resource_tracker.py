from typing import Dict
import psutil
from dataclasses import dataclass

@dataclass
class ResourceUsage:
    cpu_percent: float
    memory_used: int
    disk_io: Dict[str, int]
    network_io: Dict[str, int]

class TaskResourceTracker:
    def __init__(self):
        self.resource_history = []
        
    async def track_resources(self, task_id: str) -> ResourceUsage:
        """작업 리소스 사용량 추적"""
        usage = ResourceUsage(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_used=psutil.Process().memory_info().rss,
            disk_io=self._get_disk_io_stats(),
            network_io=self._get_network_stats()
        )
        
        self.resource_history.append({
            'task_id': task_id,
            'timestamp': time.time(),
            'usage': usage
        })
        
        return usage
