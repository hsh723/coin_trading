from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class WorkflowExecution:
    workflow_id: str
    stages: List[Dict]
    dependencies: Dict[str, List[str]]
    status: Dict[str, str]

class WorkflowOrchestrator:
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.active_workflows = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def orchestrate_workflow(self, workflow_def: Dict) -> WorkflowExecution:
        """워크플로우 오케스트레이션"""
        workflow_id = workflow_def['id']
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            stages=workflow_def['stages'],
            dependencies=self._build_dependency_graph(workflow_def),
            status={}
        )
        
        async with self.semaphore:
            self.active_workflows[workflow_id] = execution
            await self._execute_stages(execution)
            
        return execution
