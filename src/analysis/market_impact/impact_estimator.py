import numpy as np
import pandas as pd
from typing import Dict, Optional

class MarketImpactEstimator:
    def __init__(self, impact_model: str = 'square_root'):
        self.impact_model = impact_model
        self.model_params = {}
        
    def estimate_price_impact(self, 
                            order_size: float, 
                            market_data: pd.DataFrame) -> float:
        """주문 크기에 따른 가격 영향 추정"""
        avg_volume = market_data['volume'].mean()
        volatility = market_data['close'].pct_change().std()
        
        if self.impact_model == 'square_root':
            return self._square_root_impact(order_size, avg_volume, volatility)
        elif self.impact_model == 'linear':
            return self._linear_impact(order_size, avg_volume)
