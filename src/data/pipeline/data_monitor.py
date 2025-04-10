from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class PipelineMetrics:
    throughput: float
    latency: float
    error_rate: float
    queue_size: int
    resource_usage: Dict[str, float]

class PipelineMonitor:
    def __init__(self, monitor_config: Dict = None):
        self.config = monitor_config or {
            'sample_interval': 60,
            'metrics_history': 1000
        }
        self.metrics_history = []
        
    async def collect_metrics(self) -> PipelineMetrics:
        """파이프라인 메트릭스 수집"""
        current_metrics = PipelineMetrics(
            throughput=self._calculate_throughput(),
            latency=self._measure_latency(),
            error_rate=self._calculate_error_rate(),
            queue_size=self._get_queue_size(),
            resource_usage=self._get_resource_usage()
        )
        
        self.metrics_history.append(current_metrics)
        return current_metrics
