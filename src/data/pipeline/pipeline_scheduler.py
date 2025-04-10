from typing import Dict, List
import asyncio
from dataclasses import dataclass
import schedule

@dataclass
class ScheduledPipeline:
    pipeline_id: str
    schedule: str
    last_run: float
    next_run: float
    configuration: Dict

class PipelineScheduler:
    def __init__(self):
        self.scheduled_pipelines = {}
        self.running = False
        
    async def schedule_pipeline(self, pipeline_config: Dict) -> str:
        """파이프라인 스케줄링"""
        pipeline_id = pipeline_config['id']
        scheduled = ScheduledPipeline(
            pipeline_id=pipeline_id,
            schedule=pipeline_config['schedule'],
            last_run=0,
            next_run=self._calculate_next_run(pipeline_config['schedule']),
            configuration=pipeline_config
        )
        
        self.scheduled_pipelines[pipeline_id] = scheduled
        await self._schedule_execution(scheduled)
        
        return pipeline_id
