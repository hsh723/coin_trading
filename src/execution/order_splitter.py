from typing import List, Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class SplitOrder:
    size: float
    price: float
    delay: float
    urgency: str

class OrderSplitter:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_split_size': 0.01,
            'max_splits': 10,
            'time_window': 3600  # 1시간
        }
        
    async def split_order(self, order: Dict, market_data: Dict) -> List[SplitOrder]:
        """주문 분할"""
        total_size = order['size']
        volume_profile = self._analyze_volume_profile(market_data)
        optimal_splits = self._calculate_optimal_splits(total_size, volume_profile)
        
        return [
            SplitOrder(
                size=split_size,
                price=self._calculate_limit_price(order['side'], market_data),
                delay=self._calculate_delay(i, len(optimal_splits)),
                urgency=self._determine_urgency(i, len(optimal_splits))
            )
            for i, split_size in enumerate(optimal_splits)
        ]
