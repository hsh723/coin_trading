from typing import Dict, List
from dataclasses import dataclass

@dataclass
class EscalationLevel:
    level: int
    contacts: List[str]
    delay: int
    channels: List[str]

class AlertEscalationManager:
    def __init__(self, escalation_config: Dict = None):
        self.config = escalation_config or {
            'max_levels': 3,
            'base_delay': 300  # 5분
        }
        self.escalation_history = {}
        
    async def handle_escalation(self, alert: Dict) -> Dict:
        """알림 에스컬레이션 처리"""
        alert_id = alert['id']
        severity = alert['severity']
        
        if alert_id not in self.escalation_history:
            self.escalation_history[alert_id] = {
                'current_level': 0,
                'attempts': [],
                'resolved': False
            }
            
        escalation = await self._determine_escalation_level(alert)
        await self._notify_escalation_contacts(escalation, alert)
        
        return {
            'escalation_level': escalation.level,
            'next_notification': time.time() + escalation.delay
        }
