from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class Alert:
    alert_id: str
    severity: str
    message: str
    source: str
    timestamp: float
    metadata: Dict

class AlertManager:
    def __init__(self, alert_config: Dict = None):
        self.config = alert_config or {
            'channels': ['email', 'slack', 'telegram'],
            'severity_levels': ['info', 'warning', 'error', 'critical'],
            'batch_size': 10
        }
        self.alert_queue = asyncio.Queue()
        self.alert_history = []
        
    async def send_alert(self, alert: Alert) -> bool:
        """알림 전송"""
        if alert.severity not in self.config['severity_levels']:
            return False
            
        await self.alert_queue.put(alert)
        self.alert_history.append(alert)
        
        if self.alert_queue.qsize() >= self.config['batch_size']:
            await self._process_alert_batch()
            
        return True
