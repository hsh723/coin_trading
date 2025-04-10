from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AlertRule:
    rule_id: str
    condition: str
    threshold: float
    comparison: str
    metadata: Dict

class AlertRuleManager:
    def __init__(self):
        self.rules = {}
        self.rule_history = []
        
    async def evaluate_rules(self, data: Dict) -> List[str]:
        """알림 규칙 평가"""
        triggered_rules = []
        
        for rule_id, rule in self.rules.items():
            if self._evaluate_condition(data, rule):
                triggered_rules.append(rule_id)
                self._log_rule_trigger(rule_id, data)
                
        return triggered_rules
