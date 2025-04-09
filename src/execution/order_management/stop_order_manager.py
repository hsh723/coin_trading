from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class StopOrderStatus:
    order_id: str
    trigger_price: float
    stop_type: str
    status: str
    executed_price: float = None

class StopOrderManager:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'price_buffer': 0.001,
            'check_interval': 1.0
        }
        self.active_stops = {}
        
    async def place_stop_order(self, order: Dict) -> str:
        """스탑 주문 설정"""
        order_id = self._generate_order_id()
        self.active_stops[order_id] = {
            'trigger_price': order['trigger_price'],
            'stop_type': order['stop_type'],
            'quantity': order['quantity'],
            'status': 'active'
        }
        
        asyncio.create_task(self._monitor_stop_order(order_id))
        return order_id
