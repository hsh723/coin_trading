from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class LoadBalanceMetrics:
    worker_loads: Dict[str, float]
    task_distribution: Dict[str, List[str]]
    load_factor: float

class TaskLoadBalancer:
    def __init__(self, worker_config: Dict = None):
        self.worker_config = worker_config or {
            'max_workers': 5,
            'max_load_per_worker': 0.8
        }
        self.worker_loads = {}
        
    async def balance_tasks(self, tasks: List[Dict]) -> Dict[str, List[Dict]]:
        """작업 부하 분산"""
        worker_assignments = {}
        for task in sorted(tasks, key=lambda x: x.get('priority', 0), reverse=True):
            worker_id = await self._find_optimal_worker(task)
            if worker_id not in worker_assignments:
                worker_assignments[worker_id] = []
            worker_assignments[worker_id].append(task)
            
        return worker_assignments
