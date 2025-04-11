import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MicroPrice:
    micro_price: float
    micro_price_spread: float
    price_efficiency: float
    price_discovery: Dict[str, float]

class MicroPriceAnalyzer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    async def analyze_micro_price(self, order_book: Dict, trades: List[Dict]) -> MicroPrice:
        """미시가격 분석"""
        micro_price = self._calculate_micro_price(order_book)
        spread = self._calculate_micro_spread(order_book)
        
        return MicroPrice(
            micro_price=micro_price,
            micro_price_spread=spread,
            price_efficiency=self._calculate_price_efficiency(trades),
            price_discovery=self._analyze_price_discovery(order_book, trades)
        )
