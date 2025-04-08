import asyncio
from typing import Dict, Callable
import schedule
import time
from datetime import datetime

class TaskScheduler:
    def __init__(self):
        self.tasks: Dict[str, Callable] = {}
        self.running = False
        
    def add_task(self, name: str, task: Callable, schedule_at: str):
        """작업 추가"""
        self.tasks[name] = task
        if 'daily' in schedule_at:
            schedule.every().day.at(schedule_at.split()[1]).do(task)
        elif 'interval' in schedule_at:
            minutes = int(schedule_at.split()[1])
            schedule.every(minutes).minutes.do(task)
            
    async def run(self):
        """스케줄러 실행"""
        self.running = True
        while self.running:
            schedule.run_pending()
            await asyncio.sleep(1)
