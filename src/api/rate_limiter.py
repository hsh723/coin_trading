from typing import Dict
from dataclasses import dataclass
import time
import asyncio

@dataclass
class RateLimitInfo:
    requests_remaining: int
    time_window: float
    reset_time: float
    is_limited: bool

class ApiRateLimiter:
    def __init__(self, limits_config: Dict = None):
        self.config = limits_config or {
            'requests_per_second': 10,
            'requests_per_minute': 500,
            'burst_limit': 50
        }
        self.request_timestamps = []
        
    async def check_rate_limit(self, exchange_id: str) -> RateLimitInfo:
        """API 요청 속도 제한 확인"""
        current_time = time.time()
        self._clean_old_timestamps(current_time)
        
        requests_count = len(self.request_timestamps)
        is_limited = requests_count >= self.config['requests_per_second']
        
        if not is_limited:
            self.request_timestamps.append(current_time)
            
        return RateLimitInfo(
            requests_remaining=self.config['requests_per_second'] - requests_count,
            time_window=1.0,
            reset_time=self._calculate_reset_time(current_time),
            is_limited=is_limited
        )
