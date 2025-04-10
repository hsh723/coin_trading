from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class AlertMetrics:
    total_alerts: int
    active_alerts: int
    resolved_alerts: int
    alert_categories: Dict[str, int]
    avg_resolution_time: float

class AlertMetricsCollector:
    def __init__(self):
        self.metrics_history = []
        
    async def collect_metrics(self) -> AlertMetrics:
        """알림 메트릭스 수집"""
        current_time = time.time()
        active_alerts = self._count_active_alerts()
        resolved = self._count_resolved_alerts()
        
        metrics = AlertMetrics(
            total_alerts=active_alerts + resolved,
            active_alerts=active_alerts,
            resolved_alerts=resolved,
            alert_categories=self._categorize_alerts(),
            avg_resolution_time=self._calculate_avg_resolution_time()
        )
        
        self.metrics_history.append((current_time, metrics))
        return metrics
