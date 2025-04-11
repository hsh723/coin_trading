import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class OrderBookMetrics:
    bid_ask_imbalance: float
    book_pressure: float
    liquidity_score: float
    depth_analysis: Dict[str, float]

class OrderBookAnalyzer:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        
    async def analyze_order_book(self, order_book: Dict) -> OrderBookMetrics:
        """오더북 분석"""
        bids = np.array(order_book['bids'])[:self.depth_levels]
        asks = np.array(order_book['asks'])[:self.depth_levels]
        
        imbalance = self._calculate_imbalance(bids, asks)
        pressure = self._calculate_book_pressure(bids, asks)
        
        return OrderBookMetrics(
            bid_ask_imbalance=imbalance,
            book_pressure=pressure,
            liquidity_score=self._calculate_liquidity_score(bids, asks),
            depth_analysis=self._analyze_depth(bids, asks)
        )
