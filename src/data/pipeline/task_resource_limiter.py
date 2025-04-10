from typing import Dict
import psutil
from dataclasses import dataclass

@dataclass
class ResourceLimits:
    max_memory: int
    max_cpu_percent: float
    max_disk_io: int
    max_network_io: int

class ResourceLimiter:
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.resource_usage = {}
        
    async def check_resource_limits(self, task_id: str) -> bool:
        """리소스 제한 체크"""
        current_usage = {
            'memory': psutil.Process().memory_info().rss,
            'cpu': psutil.cpu_percent(),
            'disk_io': self._get_disk_io(),
            'network_io': self._get_network_io()
        }
        
        self.resource_usage[task_id] = current_usage
        
        return all([
            current_usage['memory'] <= self.limits.max_memory,
            current_usage['cpu'] <= self.limits.max_cpu_percent,
            current_usage['disk_io'] <= self.limits.max_disk_io,
            current_usage['network_io'] <= self.limits.max_network_io
        ])
