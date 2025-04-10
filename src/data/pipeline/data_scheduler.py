from typing import Dict, List
import asyncio
import schedule
from dataclasses import dataclass

@dataclass
class ScheduledTask:
    task_id: str
    interval: str
    last_run: float
    next_run: float
    status: str

class DataScheduler:
    def __init__(self, schedule_config: Dict = None):
        self.config = schedule_config or {
            'market_data': '1m',
            'analytics': '5m',
            'cleanup': '1h'
        }
        self.tasks = {}
        
    async def start_scheduler(self):
        """데이터 수집 스케줄러 시작"""
        for task_name, interval in self.config.items():
            self.tasks[task_name] = self._create_scheduled_task(task_name, interval)
            
        while True:
            await self._run_pending_tasks()
            await asyncio.sleep(1)
