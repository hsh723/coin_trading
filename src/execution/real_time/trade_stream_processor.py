from typing import Dict, List
from dataclasses import dataclass
import asyncio

@dataclass
class TradeStreamMetrics:
    buy_volume: float
    sell_volume: float
    trade_count: int
    price_impact: float
    trade_velocity: float

class TradeStreamProcessor:
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.trade_buffer = []
        
    async def process_trade_stream(self, trade: Dict) -> TradeStreamMetrics:
        """실시간 거래 스트림 처리"""
        self.trade_buffer.append(trade)
        if len(self.trade_buffer) > self.buffer_size:
            self.trade_buffer.pop(0)
            
        return TradeStreamMetrics(
            buy_volume=self._calculate_buy_volume(),
            sell_volume=self._calculate_sell_volume(),
            trade_count=len(self.trade_buffer),
            price_impact=self._calculate_price_impact(),
            trade_velocity=self._calculate_trade_velocity()
        )
