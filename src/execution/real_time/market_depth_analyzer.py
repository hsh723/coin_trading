from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class MarketDepthState:
    bid_depth: float
    ask_depth: float
    imbalance: float
    spread: float
    liquidity_score: float

class MarketDepthAnalyzer:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        
    async def analyze_depth(self, order_book: Dict) -> MarketDepthState:
        """실시간 시장 깊이 분석"""
        bid_depth = self._calculate_bid_depth(order_book['bids'])
        ask_depth = self._calculate_ask_depth(order_book['asks'])
        imbalance = self._calculate_imbalance(bid_depth, ask_depth)
        
        return MarketDepthState(
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            imbalance=imbalance,
            spread=self._calculate_spread(order_book),
            liquidity_score=self._calculate_liquidity_score(bid_depth, ask_depth)
        )
