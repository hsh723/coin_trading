from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class PerformanceMetrics:
    processing_time: float
    throughput: float
    bottlenecks: List[str]
    resource_utilization: Dict[str, float]

class PipelinePerformanceAnalyzer:
    def __init__(self, performance_config: Dict = None):
        self.config = performance_config or {
            'metrics_window': 300,  # 5분
            'performance_threshold': 0.8
        }
        
    async def analyze_performance(self, pipeline_id: str) -> PerformanceMetrics:
        """파이프라인 성능 분석"""
        stage_timings = await self._collect_stage_timings(pipeline_id)
        resource_usage = self._monitor_resource_usage()
        bottlenecks = self._identify_bottlenecks(stage_timings)
        
        return PerformanceMetrics(
            processing_time=sum(stage_timings.values()),
            throughput=self._calculate_throughput(stage_timings),
            bottlenecks=bottlenecks,
            resource_utilization=resource_usage
        )
