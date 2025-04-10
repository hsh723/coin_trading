from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class AggregatedMetrics:
    total_tasks: int
    success_rate: float
    avg_duration: float
    error_rates: Dict[str, float]
    resource_usage: Dict[str, float]

class TaskMetricsAggregator:
    def __init__(self, aggregation_window: int = 3600):  # 1시간
        self.aggregation_window = aggregation_window
        self.metrics_buffer = []
        
    async def aggregate_metrics(self, new_metrics: List[Dict]) -> AggregatedMetrics:
        """태스크 메트릭스 집계"""
        self.metrics_buffer.extend(new_metrics)
        self._cleanup_old_metrics()
        
        df = pd.DataFrame(self.metrics_buffer)
        
        return AggregatedMetrics(
            total_tasks=len(df),
            success_rate=len(df[df['status'] == 'success']) / len(df),
            avg_duration=df['duration'].mean(),
            error_rates=self._calculate_error_rates(df),
            resource_usage=self._calculate_resource_usage(df)
        )
