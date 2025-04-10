from typing import Dict, List
from dataclasses import dataclass

@dataclass
class DashboardMetrics:
    active_alerts: int
    alert_by_severity: Dict[str, int]
    response_times: Dict[str, float]
    top_sources: List[str]

class AlertDashboard:
    def __init__(self):
        self.metrics_buffer = []
        
    async def update_dashboard(self) -> DashboardMetrics:
        """대시보드 메트릭스 업데이트"""
        active = await self._count_active_alerts()
        severities = await self._group_by_severity()
        response = await self._calculate_response_times()
        sources = await self._identify_top_sources()
        
        metrics = DashboardMetrics(
            active_alerts=active,
            alert_by_severity=severities,
            response_times=response,
            top_sources=sources
        )
        
        self.metrics_buffer.append(metrics)
        return metrics
