from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class WorkerStatus:
    worker_id: str
    load: float
    active_tasks: int
    health_status: str

class TaskDistributionManager:
    def __init__(self, worker_config: Dict = None):
        self.config = worker_config or {
            'max_workers': 5,
            'max_load_per_worker': 0.8
        }
        self.workers = {}
        
    async def distribute_tasks(self, tasks: List[Dict]) -> Dict[str, List[Dict]]:
        """작업 분배"""
        distribution = {}
        worker_loads = self._get_worker_loads()
        
        for task in sorted(tasks, key=lambda x: x.get('priority', 0), reverse=True):
            worker_id = await self._find_optimal_worker(worker_loads)
            if worker_id not in distribution:
                distribution[worker_id] = []
            distribution[worker_id].append(task)
            worker_loads[worker_id] += self._calculate_task_load(task)
            
        return distribution
