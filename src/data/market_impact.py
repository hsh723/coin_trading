import numpy as np
import pandas as pd
from typing import Dict

class MarketImpactAnalyzer:
    def __init__(self, impact_threshold: float = 0.01):
        self.impact_threshold = impact_threshold
        
    def estimate_impact(self, order_size: float, market_data: pd.DataFrame) -> Dict[str, float]:
        """주문 크기의 시장 영향 추정"""
        volume = market_data['volume'].mean()
        price = market_data['close'].iloc[-1]
        impact_ratio = order_size / volume
        
        price_impact = self._calculate_price_impact(impact_ratio, price)
        return {
            'price_impact': price_impact,
            'impact_ratio': impact_ratio,
            'estimated_slippage': price_impact * order_size
        }
