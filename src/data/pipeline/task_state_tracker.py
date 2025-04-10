from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class TaskState:
    task_id: str
    status: str
    start_time: float
    end_time: float
    execution_history: List[Dict]
    retries: int

class TaskStateTracker:
    def __init__(self):
        self.task_states = {}
        self.state_transitions = []
        
    async def track_task_state(self, task_id: str, new_state: str) -> TaskState:
        """작업 상태 추적"""
        current_time = time.time()
        
        if task_id not in self.task_states:
            self.task_states[task_id] = TaskState(
                task_id=task_id,
                status=new_state,
                start_time=current_time,
                end_time=0,
                execution_history=[],
                retries=0
            )
        else:
            self.task_states[task_id].status = new_state
            self.task_states[task_id].execution_history.append({
                'timestamp': current_time,
                'state': new_state
            })
            
        return self.task_states[task_id]
