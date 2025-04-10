from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SeverityConfig:
    name: str
    level: int
    color: str
    requires_ack: bool
    auto_escalate: bool

class AlertSeverityManager:
    def __init__(self):
        self.severity_levels = {
            'critical': SeverityConfig('Critical', 1, 'red', True, True),
            'high': SeverityConfig('High', 2, 'orange', True, False),
            'medium': SeverityConfig('Medium', 3, 'yellow', False, False),
            'low': SeverityConfig('Low', 4, 'blue', False, False)
        }
        
    async def evaluate_severity(self, alert_data: Dict) -> str:
        """알림 심각도 평가"""
        base_severity = alert_data.get('severity', 'low')
        if self._check_critical_conditions(alert_data):
            return 'critical'
        elif self._check_high_severity_conditions(alert_data):
            return 'high'
            
        return base_severity
