from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ResourceAllocation:
    task_id: str
    allocated_resources: Dict[str, float]
    allocation_time: float
    expected_duration: float

class ResourceAllocator:
    def __init__(self, available_resources: Dict[str, float]):
        self.available_resources = available_resources
        self.allocations = {}
        
    async def allocate_resources(self, task: Dict) -> ResourceAllocation:
        """리소스 할당"""
        required_resources = task.get('required_resources', {})
        allocation = {}
        
        for resource, amount in required_resources.items():
            if resource in self.available_resources:
                if self.available_resources[resource] >= amount:
                    allocation[resource] = amount
                    self.available_resources[resource] -= amount
                    
        return ResourceAllocation(
            task_id=task['id'],
            allocated_resources=allocation,
            allocation_time=time.time(),
            expected_duration=task.get('estimated_duration', 0)
        )
