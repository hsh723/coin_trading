from typing import Dict, List
import logging
from dataclasses import dataclass
import time

@dataclass
class TaskLog:
    task_id: str
    timestamp: float
    level: str
    message: str
    context: Dict

class TaskLogger:
    def __init__(self, log_path: str = "pipeline_tasks.log"):
        self.logger = logging.getLogger("task_logger")
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
        
    async def log_task(self, task_id: str, level: str, 
                      message: str, context: Dict = None) -> TaskLog:
        """작업 로깅"""
        log_entry = TaskLog(
            task_id=task_id,
            timestamp=time.time(),
            level=level,
            message=message,
            context=context or {}
        )
        
        getattr(self.logger, level.lower())(
            f"Task {task_id}: {message}"
        )
        
        return log_entry
