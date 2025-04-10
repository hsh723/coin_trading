from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MarketDepthAnalysis:
    bid_ask_ratio: float
    depth_imbalance: float
    liquidity_concentration: Dict[str, float]
    order_book_pressure: float

class MarketDepthAnalyzer:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        self.depth_history = []
        
    async def analyze_depth(self, order_book: Dict) -> MarketDepthAnalysis:
        """시장 호가창 분석"""
        bids = order_book['bids'][:self.depth_levels]
        asks = order_book['asks'][:self.depth_levels]
        
        bid_ask_ratio = self._calculate_bid_ask_ratio(bids, asks)
        imbalance = self._calculate_depth_imbalance(bids, asks)
        concentration = self._analyze_liquidity_concentration(bids, asks)
        pressure = self._calculate_order_book_pressure(bids, asks)
        
        return MarketDepthAnalysis(
            bid_ask_ratio=bid_ask_ratio,
            depth_imbalance=imbalance,
            liquidity_concentration=concentration,
            order_book_pressure=pressure
        )
