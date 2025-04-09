import asyncio
from typing import Dict, List
from datetime import datetime, timedelta

class OrderScheduler:
    def __init__(self):
        self.scheduled_orders = []
        self.running = False
        
    async def schedule_order(self, order: Dict, schedule_time: datetime) -> str:
        """주문 스케줄링"""
        order_id = self._generate_order_id()
        self.scheduled_orders.append({
            'id': order_id,
            'order': order,
            'schedule_time': schedule_time,
            'status': 'scheduled'
        })
        
        await self._sort_schedule()
        return order_id
