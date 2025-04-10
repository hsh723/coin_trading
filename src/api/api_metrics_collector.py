from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class ApiMetrics:
    request_count: int
    success_rate: float
    avg_latency: float
    error_distribution: Dict[str, int]
    endpoint_stats: Dict[str, Dict]

class ApiMetricsCollector:
    def __init__(self, metrics_config: Dict = None):
        self.config = metrics_config or {
            'metrics_window': 3600,  # 1시간
            'latency_threshold': 1000  # 1초
        }
        self.metrics_history = []
        
    async def collect_metrics(self, exchange_id: str) -> ApiMetrics:
        """API 메트릭스 수집"""
        current_metrics = self._get_current_window_metrics(exchange_id)
        
        return ApiMetrics(
            request_count=len(current_metrics),
            success_rate=self._calculate_success_rate(current_metrics),
            avg_latency=self._calculate_avg_latency(current_metrics),
            error_distribution=self._analyze_errors(current_metrics),
            endpoint_stats=self._analyze_endpoints(current_metrics)
        )
