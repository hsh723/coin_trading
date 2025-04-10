from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class PerformanceStats:
    execution_latency: float
    order_throughput: float
    success_rate: float
    error_rate: float
    system_metrics: Dict[str, float]

class PerformanceTracker:
    def __init__(self, tracking_config: Dict = None):
        self.config = tracking_config or {
            'metrics_window': 1000,
            'alert_thresholds': {
                'latency': 1.0,  # 1초
                'error_rate': 0.05  # 5%
            }
        }
        self.performance_history = []
        
    async def track_performance(self, execution_data: Dict) -> PerformanceStats:
        """실행 성능 추적"""
        stats = PerformanceStats(
            execution_latency=self._calculate_latency(execution_data),
            order_throughput=self._calculate_throughput(),
            success_rate=self._calculate_success_rate(),
            error_rate=self._calculate_error_rate(),
            system_metrics=self._collect_system_metrics()
        )
        
        self.performance_history.append(stats)
        await self._check_performance_alerts(stats)
        
        return stats
