from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RoutingDecision:
    exchange: str
    expected_cost: float
    execution_strategy: str
    urgency_level: str

class RealTimeOrderRouter:
    def __init__(self, exchanges: List[Dict]):
        self.exchanges = exchanges
        self.routing_history = []
        
    async def find_best_route(self, order: Dict) -> RoutingDecision:
        """최적 실행 경로 찾기"""
        costs = {}
        for exchange in self.exchanges:
            if self._check_liquidity(exchange, order):
                cost = await self._estimate_execution_cost(exchange, order)
                costs[exchange['name']] = cost
                
        best_exchange = min(costs.items(), key=lambda x: x[1])[0]
        return self._create_routing_decision(best_exchange, order, costs[best_exchange])
