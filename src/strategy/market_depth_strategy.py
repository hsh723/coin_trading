from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class MarketDepthAnalysis:
    depth_imbalance: float
    bid_strength: float
    ask_strength: float
    liquidity_score: float
    trading_signal: str

class MarketDepthStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'depth_levels': 10,
            'imbalance_threshold': 0.2,
            'min_liquidity': 1.0
        }
        
    async def analyze_depth(self, order_book: Dict) -> MarketDepthAnalysis:
        """시장 깊이 분석"""
        bid_depth = self._calculate_bid_depth(order_book['bids'])
        ask_depth = self._calculate_ask_depth(order_book['asks'])
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        
        return MarketDepthAnalysis(
            depth_imbalance=imbalance,
            bid_strength=bid_depth,
            ask_strength=ask_depth,
            liquidity_score=self._calculate_liquidity_score(order_book),
            trading_signal=self._generate_signal(imbalance)
        )
