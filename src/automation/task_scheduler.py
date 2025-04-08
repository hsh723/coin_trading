import asyncio
from typing import Dict, Callable
import schedule
from datetime import datetime, timedelta

class TaskScheduler:
    def __init__(self):
        self.tasks = {}
        self.running = False
        
    def add_task(self, name: str, task: Callable, schedule_type: str, interval: str):
        """작업 스케줄 추가"""
        self.tasks[name] = {
            'task': task,
            'schedule': self._parse_schedule(schedule_type, interval),
            'last_run': None,
            'next_run': None
        }
        
    async def run(self):
        """스케줄러 실행"""
        self.running = True
        while self.running:
            await self._execute_due_tasks()
            await asyncio.sleep(1)
