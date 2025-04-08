from typing import Dict, List
import asyncio
from ..exchange.base import ExchangeBase

class ExchangeRouter:
    def __init__(self, exchanges: Dict[str, ExchangeBase]):
        self.exchanges = exchanges
        self.routing_rules = {}
        
    async def route_order(self, order: Dict) -> Dict:
        """최적 거래소로 주문 라우팅"""
        best_exchange = await self._find_best_exchange(order)
        return await self.exchanges[best_exchange].create_order(
            symbol=order['symbol'],
            order_type=order['type'],
            side=order['side'],
            amount=order['amount']
        )
