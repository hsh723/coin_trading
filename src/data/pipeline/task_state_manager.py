from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum

class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskStateManager:
    def __init__(self):
        self.task_states = {}
        self.state_history = []
        
    async def update_task_state(self, task_id: str, 
                              new_state: TaskState, 
                              metadata: Dict = None) -> bool:
        """작업 상태 업데이트"""
        try:
            old_state = self.task_states.get(task_id)
            self.task_states[task_id] = new_state
            
            self.state_history.append({
                'task_id': task_id,
                'old_state': old_state,
                'new_state': new_state,
                'timestamp': time.time(),
                'metadata': metadata or {}
            })
            
            return True
        except Exception as e:
            await self._handle_state_error(e, task_id)
            return False
