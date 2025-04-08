from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class OrderRoute:
    exchange: str
    price: float
    volume: float
    fee: float
    total_cost: float

class SmartOrderRouter:
    def __init__(self, exchanges: List[Dict]):
        self.exchanges = exchanges
        
    async def find_best_route(self, symbol: str, amount: float) -> OrderRoute:
        """최적 주문 경로 탐색"""
        routes = []
        for exchange in self.exchanges:
            orderbook = await self._fetch_orderbook(exchange, symbol)
            route = self._calculate_route_cost(orderbook, amount, exchange)
            if route:
                routes.append(route)
                
        return min(routes, key=lambda x: x.total_cost)
