from typing import Dict, List, Optional
import asyncio
from dataclasses import dataclass

@dataclass
class QueueMetrics:
    queue_size: int
    processing_count: int
    waiting_count: int
    average_wait_time: float

class TaskQueueManager:
    def __init__(self, max_queue_size: int = 1000):
        self.max_queue_size = max_queue_size
        self.task_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing_tasks = set()
        
    async def enqueue_task(self, task: Dict) -> bool:
        """작업 큐에 추가"""
        if self.task_queue.qsize() >= self.max_queue_size:
            return False
            
        await self.task_queue.put({
            'task': task,
            'enqueue_time': time.time(),
            'priority': task.get('priority', 0)
        })
        return True
