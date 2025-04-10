from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class AlertContext:
    source: str
    alert_type: str
    metadata: Dict
    handled_by: List[str]

class AlertHandler:
    def __init__(self):
        self.handlers = {}
        self.active_alerts = {}
        
    async def handle_alert(self, alert: Dict) -> bool:
        """알림 처리"""
        alert_id = alert.get('id')
        context = AlertContext(
            source=alert.get('source'),
            alert_type=alert.get('type'),
            metadata=alert.get('metadata', {}),
            handled_by=[]
        )
        
        handlers = self._get_relevant_handlers(alert)
        results = await asyncio.gather(*[
            handler.handle(alert, context) 
            for handler in handlers
        ])
        
        return all(results)
