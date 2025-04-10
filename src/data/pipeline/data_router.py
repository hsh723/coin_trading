from typing import Dict, List
import asyncio
from dataclasses import dataclass

@dataclass
class RoutingRule:
    source: str
    destination: str
    filter_condition: Dict
    transform_rule: Dict

class DataRouter:
    def __init__(self, routing_rules: List[RoutingRule]):
        self.routing_rules = routing_rules
        self.routes = {}
        
    async def route_data(self, data: Dict, source: str) -> Dict[str, bool]:
        """데이터 라우팅"""
        results = {}
        applicable_rules = [rule for rule in self.routing_rules 
                          if rule.source == source]
        
        for rule in applicable_rules:
            if self._check_filter_condition(data, rule.filter_condition):
                transformed_data = self._apply_transform(data, rule.transform_rule)
                success = await self._send_to_destination(
                    transformed_data, 
                    rule.destination
                )
                results[rule.destination] = success
                
        return results
