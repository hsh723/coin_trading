from typing import Dict, Callable
from dataclasses import dataclass
import asyncio

@dataclass
class RetryConfig:
    max_retries: int
    base_delay: float
    max_delay: float
    exponential: bool

class ApiRetryHandler:
    def __init__(self, retry_config: Dict = None):
        self.config = retry_config or {
            'max_retries': 3,
            'base_delay': 1.0,
            'max_delay': 10.0,
            'exponential': True
        }
        
    async def execute_with_retry(self, 
                               func: Callable, 
                               *args, **kwargs) -> Dict:
        """재시도 로직 실행"""
        retries = 0
        last_error = None
        
        while retries < self.config['max_retries']:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                retries += 1
                delay = self._calculate_delay(retries)
                await asyncio.sleep(delay)
                
        raise last_error
