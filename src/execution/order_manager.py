import asyncio
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class OrderStatus:
    order_id: str
    status: str
    filled_amount: float
    remaining_amount: float
    average_price: float

class OrderManager:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_active_orders': 10,
            'order_timeout': 60
        }
        self.active_orders: Dict[str, Dict] = {}
        
    async def place_order(self, order: Dict) -> OrderStatus:
        """주문 실행 및 관리"""
        if len(self.active_orders) >= self.config['max_active_orders']:
            await self._clean_completed_orders()
            
        order_result = await self._execute_order(order)
        self.active_orders[order_result.order_id] = order_result
        
        return order_result
