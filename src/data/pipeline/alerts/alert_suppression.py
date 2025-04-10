from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class SuppressionRule:
    rule_id: str
    pattern: str
    duration: int
    conditions: Dict

class AlertSuppressor:
    def __init__(self):
        self.suppressed_alerts = {}
        self.rules = {}
        
    async def should_suppress(self, alert: Dict) -> bool:
        """알림 억제 여부 확인"""
        alert_key = self._generate_alert_key(alert)
        
        if alert_key in self.suppressed_alerts:
            suppression_time = self.suppressed_alerts[alert_key]
            if time.time() - suppression_time < self._get_suppression_duration(alert):
                return True
                
        matching_rule = self._find_matching_rule(alert)
        if matching_rule:
            self.suppressed_alerts[alert_key] = time.time()
            return True
            
        return False
