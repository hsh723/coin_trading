from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class WorkflowState:
    workflow_id: str
    current_stage: str
    completed_stages: List[str]
    pending_stages: List[str]
    stage_results: Dict[str, any]

class StrategyWorkflowManager:
    def __init__(self, workflow_config: Dict = None):
        self.config = workflow_config or {
            'stages': [
                'market_analysis',
                'signal_generation',
                'risk_assessment',
                'execution_planning'
            ]
        }
        self.active_workflows = {}
        
    async def execute_workflow(self, strategy_id: str, 
                             market_data: Dict) -> WorkflowState:
        """전략 워크플로우 실행"""
        workflow_id = f"{strategy_id}_{int(time.time())}"
        state = WorkflowState(
            workflow_id=workflow_id,
            current_stage='market_analysis',
            completed_stages=[],
            pending_stages=self.config['stages'][1:],
            stage_results={}
        )
        
        for stage in self.config['stages']:
            result = await self._execute_stage(stage, market_data)
            state.stage_results[stage] = result
            state.completed_stages.append(stage)
            state.current_stage = stage
            
        return state
