import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MarketDepthMetrics:
    bid_ask_ratio: float
    depth_imbalance: float
    liquidity_score: float
    price_impact: Dict[str, float]

class MarketDepthAnalyzer:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        
    def analyze_depth(self, orderbook: pd.DataFrame) -> MarketDepthMetrics:
        """시장 깊이 분석"""
        bid_volume = orderbook[orderbook['side'] == 'bid']['volume'].sum()
        ask_volume = orderbook[orderbook['side'] == 'ask']['volume'].sum()
        
        return MarketDepthMetrics(
            bid_ask_ratio=bid_volume / ask_volume if ask_volume > 0 else 0,
            depth_imbalance=self._calculate_depth_imbalance(orderbook),
            liquidity_score=self._calculate_liquidity_score(orderbook),
            price_impact=self._calculate_price_impact(orderbook)
        )
