from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class ProgressStatus:
    task_id: str
    percent_complete: float
    current_stage: str
    eta: Optional[float]
    steps_completed: int

class TaskProgressTracker:
    def __init__(self):
        self.progress_data = {}
        
    async def update_progress(self, task_id: str, progress: float, 
                            stage: str) -> ProgressStatus:
        """작업 진행 상황 업데이트"""
        status = ProgressStatus(
            task_id=task_id,
            percent_complete=progress,
            current_stage=stage,
            eta=self._calculate_eta(task_id, progress),
            steps_completed=self._count_completed_steps(task_id)
        )
        
        self.progress_data[task_id] = status
        return status
