import asyncio
from typing import Dict, List
import numpy as np

class TickProcessor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'tick_buffer_size': 1000,
            'processing_interval': 0.1
        }
        self.tick_buffer = []
        
    async def process_ticks(self, tick_queue: asyncio.Queue):
        """실시간 틱 데이터 처리"""
        while True:
            tick = await tick_queue.get()
            await self._process_tick(tick)
            await self._update_metrics()
            await asyncio.sleep(self.config['processing_interval'])
