from typing import Dict, List
from dataclasses import dataclass
import heapq

@dataclass
class TaskPriority:
    task_id: str
    priority: int
    deadline: float
    resource_requirements: Dict

class TaskPrioritizer:
    def __init__(self, priority_config: Dict = None):
        self.config = priority_config or {
            'max_priority': 10,
            'deadline_weight': 0.6,
            'resource_weight': 0.4
        }
        self.priority_queue = []
        
    async def prioritize_tasks(self, tasks: List[Dict]) -> List[TaskPriority]:
        """작업 우선순위 결정"""
        prioritized_tasks = []
        
        for task in tasks:
            priority_score = self._calculate_priority_score(task)
            heapq.heappush(
                self.priority_queue,
                (-priority_score, TaskPriority(
                    task_id=task['id'],
                    priority=priority_score,
                    deadline=task.get('deadline', float('inf')),
                    resource_requirements=task.get('resources', {})
                ))
            )
        
        while self.priority_queue:
            _, task = heapq.heappop(self.priority_queue)
            prioritized_tasks.append(task)
            
        return prioritized_tasks
