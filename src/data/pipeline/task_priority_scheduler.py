from typing import Dict, List
import heapq
from dataclasses import dataclass

@dataclass
class ScheduledTask:
    task_id: str
    priority: int
    deadline: float
    estimated_duration: float
    dependencies: List[str]

class TaskPriorityScheduler:
    def __init__(self):
        self.task_queue = []
        self.scheduled_tasks = {}
        
    async def schedule_task(self, task: Dict) -> ScheduledTask:
        """우선순위 기반 작업 스케줄링"""
        scheduled_task = ScheduledTask(
            task_id=task['id'],
            priority=task.get('priority', 0),
            deadline=task.get('deadline', float('inf')),
            estimated_duration=task.get('estimated_duration', 0),
            dependencies=task.get('dependencies', [])
        )
        
        heapq.heappush(self.task_queue, 
                      (-scheduled_task.priority, scheduled_task))
        self.scheduled_tasks[scheduled_task.task_id] = scheduled_task
        
        return scheduled_task
