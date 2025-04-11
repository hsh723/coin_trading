import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MarketMakerProfile:
    participation_rate: float
    quote_duration: float
    spread_contribution: float
    inventory_metrics: Dict[str, float]

class MarketMakerProfiler:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
    async def analyze_market_maker(self, order_book: Dict, trades: List[Dict]) -> MarketMakerProfile:
        """마켓메이커 프로파일 분석"""
        participation = self._calculate_participation(order_book, trades)
        quote_time = self._analyze_quote_duration(order_book)
        
        return MarketMakerProfile(
            participation_rate=participation,
            quote_duration=quote_time,
            spread_contribution=self._calculate_spread_contribution(order_book),
            inventory_metrics=self._analyze_inventory_metrics(trades)
        )
