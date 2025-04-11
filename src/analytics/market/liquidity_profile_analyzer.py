import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class LiquidityProfile:
    weighted_spread: float
    depth_profile: Dict[str, float]
    liquidity_score: float
    imbalance_ratio: float

class LiquidityProfileAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'depth_weights': [0.4, 0.3, 0.2, 0.1],
            'min_liquidity': 1.0
        }
        
    async def analyze_liquidity(self, order_book: Dict) -> LiquidityProfile:
        """유동성 프로파일 분석"""
        weighted_spread = self._calculate_weighted_spread(order_book)
        depth = self._calculate_depth_profile(order_book)
        
        return LiquidityProfile(
            weighted_spread=weighted_spread,
            depth_profile=depth,
            liquidity_score=self._calculate_liquidity_score(weighted_spread, depth),
            imbalance_ratio=self._calculate_imbalance(order_book)
        )
