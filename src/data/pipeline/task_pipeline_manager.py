from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class PipelineStage:
    stage_id: str
    processor: callable
    input_schema: Dict
    output_schema: Dict
    retry_policy: Dict

class PipelineManager:
    def __init__(self, pipeline_config: Dict = None):
        self.config = pipeline_config or {
            'max_retries': 3,
            'timeout': 300,
            'parallel_stages': True
        }
        self.stages = {}
        self.stage_results = {}
        
    async def execute_pipeline(self, data: Dict) -> Dict:
        """파이프라인 실행"""
        stage_order = self._determine_stage_order()
        results = {}
        
        if self.config['parallel_stages']:
            tasks = [self._execute_stage(stage_id, data) 
                    for stage_id in stage_order]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            for stage_id in stage_order:
                result = await self._execute_stage(stage_id, data)
                results[stage_id] = result
                
        return self._process_results(results)
