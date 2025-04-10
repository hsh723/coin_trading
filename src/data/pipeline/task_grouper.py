from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TaskGroup:
    group_id: str
    tasks: List[str]
    total_weight: float
    estimated_duration: float
    dependencies: List[str]

class TaskGroupManager:
    def __init__(self, grouping_config: Dict = None):
        self.config = grouping_config or {
            'max_group_size': 5,
            'max_group_weight': 10.0
        }
        self.groups = {}
        
    async def create_task_groups(self, tasks: List[Dict]) -> List[TaskGroup]:
        """작업 그룹화"""
        current_group = []
        current_weight = 0.0
        groups = []
        
        for task in sorted(tasks, key=lambda x: x.get('weight', 1.0)):
            if len(current_group) >= self.config['max_group_size'] or \
               current_weight + task.get('weight', 1.0) > self.config['max_group_weight']:
                groups.append(self._create_group(current_group))
                current_group = []
                current_weight = 0.0
                
            current_group.append(task)
            current_weight += task.get('weight', 1.0)
            
        if current_group:
            groups.append(self._create_group(current_group))
            
        return groups
