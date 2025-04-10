from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class FailureInfo:
    task_id: str
    error_type: str
    error_message: str
    retry_count: int
    recovery_action: str

class TaskFailureHandler:
    def __init__(self, retry_config: Dict = None):
        self.retry_config = retry_config or {
            'max_retries': 3,
            'retry_delay': 5,
            'exponential_backoff': True
        }
        
    async def handle_failure(self, task_id: str, error: Exception) -> Optional[str]:
        """작업 실패 처리"""
        failure_info = FailureInfo(
            task_id=task_id,
            error_type=type(error).__name__,
            error_message=str(error),
            retry_count=0,
            recovery_action='retry'
        )
        
        return await self._execute_recovery_strategy(failure_info)
