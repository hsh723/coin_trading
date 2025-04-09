from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class OrderBookState:
    timestamp: float
    best_bid: float
    best_ask: float
    bid_depth: Dict[float, float]
    ask_depth: Dict[float, float]
    imbalance_ratio: float

class OrderBookMonitor:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        self.order_books = {}
        
    async def monitor_order_book(self, symbol: str):
        """실시간 오더북 모니터링"""
        while True:
            try:
                order_book = await self._fetch_order_book(symbol)
                state = self._analyze_order_book(order_book)
                await self._process_state(symbol, state)
                await self._check_anomalies(state)
                await asyncio.sleep(0.1)  # 100ms 간격으로 업데이트
            except Exception as e:
                self._handle_error(symbol, e)
