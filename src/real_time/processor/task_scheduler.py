import asyncio
from typing import Dict, List
from dataclasses import dataclass
import heapq

@dataclass
class TaskSchedule:
    priority: int
    task_id: str
    execution_time: float
    resources: Dict[str, float]

class TaskScheduler:
    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.task_queue = []
        self.running_tasks = {}
        
    async def schedule_tasks(self, tasks: List[Dict]) -> Dict:
        """작업 스케줄링"""
        for task in tasks:
            heapq.heappush(self.task_queue, (task['priority'], task))
            
        scheduled = await self._process_task_queue()
        return {
            'scheduled_tasks': scheduled,
            'queue_length': len(self.task_queue),
            'running_tasks': len(self.running_tasks),
            'next_execution': self._get_next_execution_time()
        }
