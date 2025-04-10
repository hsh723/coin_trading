from typing import Dict, List, Optional
from dataclasses import dataclass
import time

@dataclass
class TaskStatus:
    task_id: str
    current_status: str
    last_updated: float
    status_history: List[Dict]
    error_count: int
    retry_count: int
    
class TaskStatusManager:
    def __init__(self):
        self.task_statuses = {}
        self.status_transitions = {
            'pending': ['running', 'cancelled'],
            'running': ['completed', 'failed', 'paused'],
            'paused': ['running', 'cancelled'],
            'failed': ['pending', 'cancelled'],
            'completed': [],
            'cancelled': []
        }
        
    async def update_status(self, task_id: str, new_status: str) -> Optional[TaskStatus]:
        """작업 상태 업데이트"""
        if task_id not in self.task_statuses:
            status = TaskStatus(
                task_id=task_id,
                current_status=new_status,
                last_updated=time.time(),
                status_history=[],
                error_count=0,
                retry_count=0
            )
            self.task_statuses[task_id] = status
            return status
            
        current = self.task_statuses[task_id]
        if new_status in self.status_transitions[current.current_status]:
            current.status_history.append({
                'from': current.current_status,
                'to': new_status,
                'timestamp': time.time()
            })
            current.current_status = new_status
            current.last_updated = time.time()
            return current
            
        return None
