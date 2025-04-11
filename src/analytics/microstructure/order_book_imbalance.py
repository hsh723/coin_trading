import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class OrderBookImbalance:
    imbalance_ratio: float
    price_pressure: float
    depth_asymmetry: float
    liquidity_bias: str

class OrderBookImbalanceAnalyzer:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        
    async def analyze_imbalance(self, order_book: Dict) -> OrderBookImbalance:
        """오더북 불균형 분석"""
        bids = np.array(order_book['bids'])[:self.depth_levels]
        asks = np.array(order_book['asks'])[:self.depth_levels]
        
        imbalance = self._calculate_imbalance(bids, asks)
        pressure = self._calculate_price_pressure(bids, asks)
        
        return OrderBookImbalance(
            imbalance_ratio=imbalance,
            price_pressure=pressure,
            depth_asymmetry=self._calculate_depth_asymmetry(bids, asks),
            liquidity_bias='buy' if imbalance > 0 else 'sell'
        )
