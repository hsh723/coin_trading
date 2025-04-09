from typing import Dict
import numpy as np
from collections import deque

class OrderFlowTracker:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.trade_buffer = deque(maxlen=window_size)
        
    async def track_order_flow(self, trade: Dict) -> Dict:
        """실시간 주문 흐름 추적"""
        self.trade_buffer.append(trade)
        return {
            'buy_pressure': self._calculate_buy_pressure(),
            'sell_pressure': self._calculate_sell_pressure(),
            'flow_imbalance': self._calculate_imbalance(),
            'trade_direction': self._determine_trade_direction(),
            'significant_levels': self._find_significant_levels()
        }
