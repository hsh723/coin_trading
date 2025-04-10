from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class PerformanceMetrics:
    response_time: float
    success_rate: float
    error_rate: float
    throughput: float
    endpoint_stats: Dict[str, Dict]

class ApiPerformanceMonitor:
    def __init__(self, monitor_config: Dict = None):
        self.config = monitor_config or {
            'metrics_window': 3600,  # 1시간
            'latency_threshold': 1.0  # 1초
        }
        self.metrics_history = []
        
    async def monitor_performance(self, exchange_id: str) -> PerformanceMetrics:
        """API 성능 모니터링"""
        current_metrics = self._collect_current_metrics(exchange_id)
        self.metrics_history.append(current_metrics)
        
        return PerformanceMetrics(
            response_time=self._calculate_avg_response_time(),
            success_rate=self._calculate_success_rate(),
            error_rate=self._calculate_error_rate(),
            throughput=self._calculate_throughput(),
            endpoint_stats=self._analyze_endpoint_performance()
        )
