from typing import Dict, List
from dataclasses import dataclass
import time
import asyncio

@dataclass
class PipelineStats:
    pipeline_id: str
    stage_metrics: Dict[str, Dict]
    throughput: float
    latency: float
    error_rate: float

class PipelineMonitor:
    def __init__(self, monitoring_config: Dict = None):
        self.config = monitoring_config or {
            'stats_window': 300,  # 5분
            'alert_threshold': 0.1
        }
        self.stats_buffer = {}
        
    async def monitor_pipeline(self, pipeline_id: str) -> PipelineStats:
        """파이프라인 모니터링"""
        current_time = time.time()
        stats = self._collect_stage_metrics(pipeline_id)
        
        pipeline_stats = PipelineStats(
            pipeline_id=pipeline_id,
            stage_metrics=stats,
            throughput=self._calculate_throughput(stats),
            latency=self._calculate_latency(stats),
            error_rate=self._calculate_error_rate(stats)
        )
        
        await self._check_alerts(pipeline_stats)
        self._update_stats_buffer(pipeline_id, pipeline_stats)
        
        return pipeline_stats
