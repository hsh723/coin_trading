from typing import Dict, List
import asyncio
from dataclasses import dataclass
import heapq

@dataclass
class QueuedTask:
    task_id: str
    priority: int
    execution_time: float
    payload: Dict

class TaskQueueManager:
    def __init__(self, queue_size: int = 1000):
        self.queue_size = queue_size
        self.task_queue = []
        self.processing = set()
        
    async def enqueue_task(self, task: QueuedTask) -> bool:
        """작업 큐에 추가"""
        if len(self.task_queue) >= self.queue_size:
            return False
            
        heapq.heappush(
            self.task_queue, 
            (task.priority, task.execution_time, task)
        )
        return True
