from typing import Dict, Optional
import asyncio
from ..exchange.base import ExchangeBase

class OrderExecutor:
    def __init__(self, exchange: ExchangeBase):
        self.exchange = exchange
        self.execution_queue = asyncio.Queue()
        
    async def execute_order(self, order: Dict) -> Dict:
        """주문 실행"""
        await self.execution_queue.put(order)
        return await self._process_order(order)
        
    async def _process_order(self, order: Dict) -> Dict:
        """주문 처리 및 모니터링"""
        try:
            result = await self.exchange.create_order(
                symbol=order['symbol'],
                order_type=order['type'],
                side=order['side'],
                amount=order['amount'],
                price=order.get('price')
            )
            return await self._monitor_order(result['id'])
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
