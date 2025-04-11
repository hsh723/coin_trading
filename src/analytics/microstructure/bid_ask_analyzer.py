import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class BidAskMetrics:
    effective_spread: float
    quoted_spread: float
    spread_stability: float
    market_quality: float

class BidAskAnalyzer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    async def analyze_spreads(self, order_book: Dict) -> BidAskMetrics:
        """매수-매도 스프레드 분석"""
        return BidAskMetrics(
            effective_spread=self._calculate_effective_spread(order_book),
            quoted_spread=self._calculate_quoted_spread(order_book),
            spread_stability=self._analyze_spread_stability(order_book),
            market_quality=self._calculate_market_quality(order_book)
        )
