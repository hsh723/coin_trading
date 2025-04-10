from typing import Dict, List
from dataclasses import dataclass

@dataclass
class LiquidityMetrics:
    overall_liquidity: float
    bid_liquidity: float
    ask_liquidity: float
    depth_profile: Dict[str, float]
    liquidity_score: float

class LiquidityAnalyzer:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        
    async def analyze_liquidity(self, order_book: Dict) -> LiquidityMetrics:
        """유동성 분석"""
        bid_liquidity = self._calculate_bid_liquidity(order_book['bids'])
        ask_liquidity = self._calculate_ask_liquidity(order_book['asks'])
        
        return LiquidityMetrics(
            overall_liquidity=(bid_liquidity + ask_liquidity) / 2,
            bid_liquidity=bid_liquidity,
            ask_liquidity=ask_liquidity,
            depth_profile=self._calculate_depth_profile(order_book),
            liquidity_score=self._calculate_liquidity_score(bid_liquidity, ask_liquidity)
        )
