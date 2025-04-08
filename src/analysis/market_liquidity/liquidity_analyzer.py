import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class LiquidityMetrics:
    bid_ask_spread: float
    market_depth: float
    resiliency: float
    tightness: float
    immediacy: float

class LiquidityAnalyzer:
    def __init__(self, depth_threshold: float = 0.1):
        self.depth_threshold = depth_threshold
        
    def analyze_liquidity(self, orderbook: pd.DataFrame, trades: pd.DataFrame) -> LiquidityMetrics:
        """시장 유동성 분석"""
        spread = self._calculate_spread(orderbook)
        depth = self._calculate_market_depth(orderbook)
        resiliency = self._calculate_resiliency(trades)
        
        return LiquidityMetrics(
            bid_ask_spread=spread,
            market_depth=depth,
            resiliency=resiliency,
            tightness=spread/orderbook['mid_price'].mean(),
            immediacy=self._calculate_immediacy(trades)
        )
