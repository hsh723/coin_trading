from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class HealthStatus:
    pipeline_id: str
    status: str
    issues: List[Dict]
    metrics: Dict[str, float]
    last_check_time: float

class PipelineHealthChecker:
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.health_history = {}
        
    async def check_pipeline_health(self, pipeline_id: str) -> HealthStatus:
        """파이프라인 상태 점검"""
        current_metrics = await self._collect_pipeline_metrics(pipeline_id)
        issues = self._identify_issues(current_metrics)
        
        status = HealthStatus(
            pipeline_id=pipeline_id,
            status='healthy' if not issues else 'unhealthy',
            issues=issues,
            metrics=current_metrics,
            last_check_time=time.time()
        )
        
        self.health_history[pipeline_id] = status
        await self._handle_health_status(status)
        
        return status
