from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RoutingConfig:
    route_id: str
    conditions: List[Dict]
    destinations: List[str]
    priority: int

class AlertRoutingManager:
    def __init__(self):
        self.routes = {}
        self.route_history = []
        
    async def route_alert(self, alert: Dict) -> List[str]:
        """알림 라우팅 결정"""
        destinations = []
        
        for route_id, route in self.routes.items():
            if self._matches_conditions(alert, route.conditions):
                destinations.extend(route.destinations)
                self._log_routing(alert, route_id)
                
        return list(set(destinations))  # 중복 제거
