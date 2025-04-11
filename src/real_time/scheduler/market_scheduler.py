import asyncio
from typing import Dict, List
import numpy as np

class MarketScheduler:
    def __init__(self):
        self.market_tasks = {}
        self.scheduled_updates = []
        self.active_subscriptions = set()
        
    async def schedule_market_updates(self, symbols: List[str]) -> None:
        for symbol in symbols:
            if symbol not in self.active_subscriptions:
                task = asyncio.create_task(self._market_update_loop(symbol))
                self.market_tasks[symbol] = task
                self.active_subscriptions.add(symbol)
