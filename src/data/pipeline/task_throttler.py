from typing import Dict, Optional
import time
import asyncio

class TaskThrottler:
    def __init__(self, rate_limits: Dict[str, int]):
        self.rate_limits = rate_limits
        self.execution_times = {}
        
    async def throttle_task(self, task_type: str) -> bool:
        """작업 실행 속도 제어"""
        if task_type not in self.rate_limits:
            return True
            
        current_time = time.time()
        last_execution = self.execution_times.get(task_type, 0)
        min_interval = 1.0 / self.rate_limits[task_type]
        
        if current_time - last_execution < min_interval:
            await asyncio.sleep(min_interval - (current_time - last_execution))
            
        self.execution_times[task_type] = time.time()
        return True
