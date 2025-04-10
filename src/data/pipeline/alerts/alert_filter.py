from typing import Dict, List
from dataclasses import dataclass

@dataclass
class FilterRule:
    rule_id: str
    field: str
    condition: str
    value: any
    priority: int

class AlertFilter:
    def __init__(self, filter_rules: List[FilterRule] = None):
        self.rules = filter_rules or []
        self.filtered_alerts = []
        
    async def filter_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """알림 필터링"""
        filtered = []
        
        for alert in alerts:
            if self._passes_filters(alert):
                filtered.append(alert)
            else:
                self.filtered_alerts.append({
                    'alert': alert,
                    'timestamp': time.time(),
                    'reason': 'Filtered by rules'
                })
                
        return filtered
