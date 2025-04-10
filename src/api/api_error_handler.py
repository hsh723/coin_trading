from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class ErrorInfo:
    error_code: str
    message: str
    timestamp: float
    retry_after: Optional[int]
    context: Dict

class ApiErrorHandler:
    def __init__(self, error_config: Dict = None):
        self.config = error_config or {
            'max_retries': 3,
            'retry_delay': 1.0,
            'error_log_path': 'logs/api_errors.log'
        }
        self.error_history = []
        
    async def handle_error(self, error: Exception, context: Dict) -> ErrorInfo:
        """API 에러 처리"""
        error_info = ErrorInfo(
            error_code=self._get_error_code(error),
            message=str(error),
            timestamp=time.time(),
            retry_after=self._get_retry_after(error),
            context=context
        )
        
        self.error_history.append(error_info)
        await self._log_error(error_info)
        await self._notify_if_critical(error_info)
        
        return error_info
