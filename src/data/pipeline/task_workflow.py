from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class WorkflowStage:
    stage_id: str
    tasks: List[Dict]
    dependencies: List[str]
    status: str

class WorkflowManager:
    def __init__(self, workflow_config: Dict = None):
        self.config = workflow_config or {
            'max_parallel_stages': 3,
            'retry_failed_stages': True
        }
        self.stages = {}
        
    async def execute_workflow(self, workflow_def: Dict) -> Dict[str, str]:
        """워크플로우 실행"""
        stage_results = {}
        
        for stage in workflow_def['stages']:
            if await self._check_dependencies(stage):
                stage_result = await self._execute_stage(stage)
                stage_results[stage['id']] = stage_result['status']
                
                if stage_result['status'] == 'failed' and not self.config['retry_failed_stages']:
                    break
                    
        return stage_results
