from typing import Dict, List
from dataclasses import dataclass
import logging
import traceback

@dataclass
class DebugInfo:
    task_id: str
    execution_trace: List[str]
    variables: Dict
    error_info: Dict = None

class TaskDebugger:
    def __init__(self, debug_level: str = 'INFO'):
        self.logger = logging.getLogger('task_debugger')
        self.logger.setLevel(debug_level)
        self.debug_history = []
        
    async def debug_task(self, task_id: str, task_context: Dict) -> DebugInfo:
        """작업 디버깅 정보 수집"""
        try:
            execution_trace = self._collect_execution_trace()
            variables = self._capture_variables(task_context)
            
            debug_info = DebugInfo(
                task_id=task_id,
                execution_trace=execution_trace,
                variables=variables
            )
            
            self.debug_history.append(debug_info)
            return debug_info
            
        except Exception as e:
            self._handle_debug_error(e, task_id)
