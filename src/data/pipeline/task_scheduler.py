from typing import Dict, List
from dataclasses import dataclass
import schedule
import time
import asyncio

@dataclass
class ScheduledTask:
    task_id: str
    schedule: str
    last_run: float
    next_run: float
    is_running: bool

class TaskScheduler:
    def __init__(self, schedule_config: Dict = None):
        self.config = schedule_config or {
            'max_concurrent': 5,
            'retry_limit': 3
        }
        self.tasks = {}
        self.semaphore = asyncio.Semaphore(self.config['max_concurrent'])
        
    async def schedule_task(self, task_id: str, interval: str, 
                          task_func: callable) -> ScheduledTask:
        """작업 스케줄링"""
        scheduled_task = ScheduledTask(
            task_id=task_id,
            schedule=interval,
            last_run=0,
            next_run=self._calculate_next_run(interval),
            is_running=False
        )
        
        self.tasks[task_id] = scheduled_task
        await self._schedule_execution(task_id, task_func)
        
        return scheduled_task
