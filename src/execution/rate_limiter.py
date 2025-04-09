import asyncio
import time
from typing import Dict
from dataclasses import dataclass

@dataclass
class RateLimitConfig:
    requests_per_second: int
    requests_per_minute: int
    requests_per_hour: int

class RateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_history = []
        
    async def acquire(self, weight: int = 1) -> bool:
        """API 요청 가능 여부 확인"""
        current_time = time.time()
        self._clean_old_requests(current_time)
        
        if self._check_limits(weight):
            self.request_history.append((current_time, weight))
            return True
        
        await self._wait_for_capacity(weight)
        return True
