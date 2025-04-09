import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class LiquidityMetrics:
    bid_ask_spread: float
    depth_score: float
    impact_cost: float
    urgency_cost: float

class LiquidityAnalyzer:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        
    def analyze_execution_liquidity(self, order_book: pd.DataFrame) -> LiquidityMetrics:
        """실행 전 유동성 분석"""
        bid_ask_spread = self._calculate_spread(order_book)
        depth = self._analyze_market_depth(order_book)
        
        return LiquidityMetrics(
            bid_ask_spread=bid_ask_spread,
            depth_score=self._calculate_depth_score(depth),
            impact_cost=self._estimate_impact_cost(depth),
            urgency_cost=self._calculate_urgency_cost(bid_ask_spread, depth)
        )
