from typing import Dict, List
import asyncio
from dataclasses import dataclass

@dataclass
class QueueStatus:
    pending_orders: int
    processing_orders: int
    completed_orders: int
    queue_health: str

class OrderQueueManager:
    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self.order_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processing = set()
        
    async def add_order(self, order: Dict) -> bool:
        """주문 큐에 추가"""
        if self.order_queue.qsize() >= self.max_queue_size:
            return False
        
        await self.order_queue.put(order)
        return True
