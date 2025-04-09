from typing import Dict
import numpy as np
from collections import deque

class TickDataProcessor:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.tick_buffer = deque(maxlen=window_size)
        self.last_tick = None
        
    async def process_tick(self, tick: Dict) -> Dict:
        """틱 데이터 실시간 처리"""
        self.tick_buffer.append(tick)
        processed_data = {
            'micro_price': self._calculate_micro_price(),
            'tick_direction': self._determine_tick_direction(tick),
            'spread': self._calculate_spread(tick),
            'trade_flow': self._analyze_trade_flow()
        }
        
        self.last_tick = tick
        return processed_data
