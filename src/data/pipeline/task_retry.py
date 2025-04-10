from typing import Dict, Callable
import asyncio
from dataclasses import dataclass

@dataclass
class RetryStats:
    attempt_count: int
    last_error: str
    next_retry_time: float
    backoff_duration: float

class TaskRetryManager:
    def __init__(self, retry_config: Dict = None):
        self.config = retry_config or {
            'max_retries': 3,
            'initial_delay': 1,
            'backoff_factor': 2
        }
        self.retry_stats = {}
        
    async def retry_task(self, task_id: str, 
                        task_func: Callable, 
                        *args, **kwargs) -> Dict:
        """작업 재시도 실행"""
        attempt = 0
        delay = self.config['initial_delay']
        
        while attempt < self.config['max_retries']:
            try:
                result = await task_func(*args, **kwargs)
                return {'success': True, 'result': result}
            except Exception as e:
                attempt += 1
                if attempt == self.config['max_retries']:
                    return {'success': False, 'error': str(e)}
                    
                delay *= self.config['backoff_factor']
                await asyncio.sleep(delay)
