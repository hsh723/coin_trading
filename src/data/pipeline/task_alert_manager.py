from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class Alert:
    severity: str
    message: str
    timestamp: float
    task_id: str
    context: Dict

class TaskAlertManager:
    def __init__(self, alert_config: Dict = None):
        self.config = alert_config or {
            'max_alerts': 100,
            'alert_channels': ['email', 'slack']
        }
        self.alerts = []
        
    async def send_alert(self, alert: Alert) -> bool:
        """알림 전송"""
        try:
            for channel in self.config['alert_channels']:
                await self._send_to_channel(channel, alert)
            
            self.alerts.append(alert)
            if len(self.alerts) > self.config['max_alerts']:
                self.alerts.pop(0)
                
            return True
        except Exception as e:
            await self._handle_alert_error(e)
            return False
