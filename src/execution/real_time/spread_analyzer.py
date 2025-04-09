from typing import Dict
from dataclasses import dataclass

@dataclass
class SpreadMetrics:
    current_spread: float
    relative_spread: float
    spread_trend: str
    liquidity_score: float

class SpreadAnalyzer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.spread_history = []
        
    async def analyze_spread(self, order_book: Dict) -> SpreadMetrics:
        """실시간 스프레드 분석"""
        current_spread = order_book['asks'][0][0] - order_book['bids'][0][0]
        self.spread_history.append(current_spread)
        
        if len(self.spread_history) > self.window_size:
            self.spread_history.pop(0)
            
        avg_spread = sum(self.spread_history) / len(self.spread_history)
        
        return SpreadMetrics(
            current_spread=current_spread,
            relative_spread=current_spread / avg_spread,
            spread_trend=self._determine_trend(),
            liquidity_score=self._calculate_liquidity_score()
        )
