from typing import Dict, List
import asyncio
from dataclasses import dataclass

@dataclass
class TaskExecutionStatus:
    task_id: str
    status: str
    start_time: float
    end_time: float
    error: str = None

class TaskOrchestrator:
    def __init__(self, max_concurrent_tasks: int = 5):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.task_queue = asyncio.Queue()
        
    async def orchestrate_tasks(self, task_list: List[Dict]) -> List[TaskExecutionStatus]:
        """작업 실행 조정"""
        task_statuses = []
        for task in task_list:
            await self.task_queue.put(task)
            
        workers = [self._task_worker() for _ in range(self.max_concurrent_tasks)]
        results = await asyncio.gather(*workers)
        
        return [status for worker_result in results for status in worker_result]
