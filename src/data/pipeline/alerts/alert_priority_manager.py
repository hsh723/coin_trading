from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PriorityLevel:
    level: int
    name: str
    response_time: int  # seconds
    auto_escalate: bool

class AlertPriorityManager:
    def __init__(self):
        self.priority_levels = {
            'highest': PriorityLevel(1, 'Highest', 300, True),
            'high': PriorityLevel(2, 'High', 900, True),
            'medium': PriorityLevel(3, 'Medium', 1800, False),
            'low': PriorityLevel(4, 'Low', 3600, False)
        }
        
    async def determine_priority(self, alert: Dict) -> str:
        """알림 우선순위 결정"""
        if self._is_critical_condition(alert):
            return 'highest'
        elif self._is_system_error(alert):
            return 'high'
        elif self._is_performance_issue(alert):
            return 'medium'
            
        return 'low'
