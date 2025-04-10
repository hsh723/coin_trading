from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AlertPolicy:
    policy_id: str
    conditions: List[Dict]
    actions: List[Dict]
    cooldown_period: int
    priority: int

class AlertPolicyManager:
    def __init__(self):
        self.policies = {}
        self.last_triggered = {}
        
    async def evaluate_policies(self, event: Dict) -> List[Dict]:
        """알림 정책 평가"""
        triggered_actions = []
        
        for policy_id, policy in self.policies.items():
            if self._should_evaluate_policy(policy_id):
                if self._check_conditions(event, policy.conditions):
                    actions = await self._execute_policy_actions(policy)
                    triggered_actions.extend(actions)
                    self.last_triggered[policy_id] = time.time()
                    
        return triggered_actions
