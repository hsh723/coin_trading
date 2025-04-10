from typing import Dict, Optional
import logging
from dataclasses import dataclass

@dataclass
class ErrorRecord:
    error_type: str
    message: str
    timestamp: float
    context: Dict
    severity: str

class PipelineErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history = []
        
    async def handle_error(self, error: Exception, 
                          context: Dict) -> Optional[ErrorRecord]:
        """파이프라인 에러 처리"""
        error_record = ErrorRecord(
            error_type=type(error).__name__,
            message=str(error),
            timestamp=time.time(),
            context=context,
            severity=self._determine_severity(error)
        )
        
        self.error_history.append(error_record)
        await self._notify_if_critical(error_record)
        
        return error_record
