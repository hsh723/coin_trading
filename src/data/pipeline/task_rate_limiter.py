from typing import Dict
import time
import asyncio
from dataclasses import dataclass

@dataclass
class RateLimit:
    requests_per_second: int
    burst_limit: int
    window_size: float

class TaskRateLimiter:
    def __init__(self, rate_limits: Dict[str, RateLimit]):
        self.rate_limits = rate_limits
        self.request_timestamps = {}
        
    async def acquire_permit(self, task_type: str) -> bool:
        """작업 실행 허가 획득"""
        if task_type not in self.rate_limits:
            return True
            
        limit = self.rate_limits[task_type]
        now = time.time()
        
        # 최근 요청 기록 정리
        self.request_timestamps[task_type] = [
            ts for ts in self.request_timestamps.get(task_type, [])
            if now - ts <= limit.window_size
        ]
        
        if len(self.request_timestamps[task_type]) < limit.requests_per_second:
            self.request_timestamps[task_type].append(now)
            return True
            
        return False
