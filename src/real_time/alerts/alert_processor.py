import asyncio
from typing import Dict, List

class AlertProcessor:
    def __init__(self, alert_config: Dict = None):
        self.config = alert_config or {
            'alert_threshold': 0.05,
            'check_interval': 1.0
        }
        self.active_alerts = set()
        
    async def process_alerts(self, market_data: Dict) -> List[Dict]:
        """실시간 알림 처리"""
        alerts = []
        
        if price_alert := await self._check_price_alerts(market_data):
            alerts.append(price_alert)
            
        if volume_alert := await self._check_volume_alerts(market_data):
            alerts.append(volume_alert)
            
        return alerts
