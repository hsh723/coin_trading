from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class AlertGroup:
    group_id: str
    alerts: List[Dict]
    severity: str
    timestamp: float
    summary: str

class AlertAggregator:
    def __init__(self, aggregation_window: int = 300):  # 5분
        self.aggregation_window = aggregation_window
        self.alert_buffer = []
        
    async def aggregate_alerts(self, new_alerts: List[Dict]) -> List[AlertGroup]:
        """알림 집계"""
        self.alert_buffer.extend(new_alerts)
        self._clean_old_alerts()
        
        grouped_alerts = {}
        for alert in self.alert_buffer:
            group_key = self._generate_group_key(alert)
            if group_key not in grouped_alerts:
                grouped_alerts[group_key] = []
            grouped_alerts[group_key].append(alert)
            
        return [
            AlertGroup(
                group_id=key,
                alerts=alerts,
                severity=self._determine_group_severity(alerts),
                timestamp=time.time(),
                summary=self._generate_summary(alerts)
            )
            for key, alerts in grouped_alerts.items()
        ]
