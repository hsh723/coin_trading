import logging
from typing import Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class LogEntry:
    task_id: str
    level: str
    message: str
    timestamp: float
    context: Dict

class TaskLoggingSystem:
    def __init__(self, log_config: Dict = None):
        self.config = log_config or {
            'log_level': 'INFO',
            'log_format': '%(asctime)s - %(levelname)s - %(message)s',
            'file_path': 'logs/task_pipeline.log'
        }
        self.setup_logger()
        
    def setup_logger(self):
        """로거 설정"""
        self.logger = logging.getLogger('task_pipeline')
        self.logger.setLevel(self.config['log_level'])
        
        file_handler = logging.FileHandler(self.config['file_path'])
        file_handler.setFormatter(logging.Formatter(self.config['log_format']))
        self.logger.addHandler(file_handler)
        
    async def log_task_event(self, task_id: str, level: str, 
                           message: str, context: Dict = None) -> LogEntry:
        """작업 이벤트 로깅"""
        entry = LogEntry(
            task_id=task_id,
            level=level,
            message=message,
            timestamp=time.time(),
            context=context or {}
        )
        
        getattr(self.logger, level.lower())(
            f"Task {task_id}: {message}"
        )
        
        return entry
