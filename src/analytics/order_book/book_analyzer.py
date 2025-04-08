from typing import Dict, List
import numpy as np
import pandas as pd

class OrderBookAnalyzer:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        
    async def analyze_order_book(self, order_book: Dict) -> Dict:
        """오더북 분석"""
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'volume'])
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'volume'])
        
        return {
            'bid_ask_spread': self._calculate_spread(bids, asks),
            'depth_imbalance': self._calculate_depth_imbalance(bids, asks),
            'price_impact': self._estimate_price_impact(bids, asks),
            'liquidity_score': self._calculate_liquidity_score(bids, asks)
        }
