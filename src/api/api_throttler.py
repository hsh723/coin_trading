from typing import Dict
from dataclasses import dataclass
import time
import asyncio

@dataclass
class ThrottleConfig:
    requests_per_second: int
    burst_limit: int
    window_size: float

class ApiThrottler:
    def __init__(self, throttle_config: Dict = None):
        self.config = throttle_config or {
            'requests_per_second': 10,
            'burst_limit': 20,
            'window_size': 1.0
        }
        self.request_timestamps = []
        
    async def throttle(self, exchange_id: str) -> bool:
        """API 요청 스로틀링"""
        current_time = time.time()
        window_start = current_time - self.config['window_size']
        
        # 오래된 타임스탬프 제거
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > window_start]
        
        if len(self.request_timestamps) >= self.config['requests_per_second']:
            delay = self._calculate_delay()
            await asyncio.sleep(delay)
            
        self.request_timestamps.append(current_time)
        return True
