import numpy as np
import pandas as pd
from typing import Dict, List

class LiquidityAnalyzer:
    def __init__(self, depth_levels: int = 10):
        self.depth_levels = depth_levels
        
    def analyze_orderbook_liquidity(self, orderbook: pd.DataFrame) -> Dict:
        """오더북 유동성 분석"""
        return {
            'bid_ask_spread': self._calculate_spread(orderbook),
            'depth_imbalance': self._calculate_depth_imbalance(orderbook),
            'liquidity_score': self._calculate_liquidity_score(orderbook)
        }
        
    def _calculate_slippage(self, size: float, side: str, orderbook: pd.DataFrame) -> float:
        """예상 슬리피지 계산"""
        levels = orderbook[orderbook['side'] == side].sort_values('price')
        remaining_size = size
        weighted_price = 0
