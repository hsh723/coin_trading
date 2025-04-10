from typing import Dict
import asyncio
from dataclasses import dataclass

@dataclass
class TimeoutConfig:
    default_timeout: int
    grace_period: int
    max_extensions: int

class TaskTimeoutManager:
    def __init__(self, timeout_config: TimeoutConfig):
        self.config = timeout_config
        self.active_timers = {}
        
    async def monitor_task_timeout(self, task_id: str, timeout: int = None) -> None:
        """작업 타임아웃 모니터링"""
        timeout = timeout or self.config.default_timeout
        
        try:
            async with asyncio.timeout(timeout):
                await self._monitor_execution(task_id)
        except asyncio.TimeoutError:
            await self._handle_timeout(task_id)
