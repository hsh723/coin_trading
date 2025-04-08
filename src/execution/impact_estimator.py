import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class ImpactEstimate:
    price_impact: float
    volume_impact: float
    spread_impact: float
    total_impact: float

class MarketImpactEstimator:
    def __init__(self, impact_model: str = 'square_root'):
        self.impact_model = impact_model
        
    def estimate_impact(self, order_size: float, market_data: Dict) -> ImpactEstimate:
        """시장 영향도 추정"""
        avg_volume = self._calculate_avg_volume(market_data)
        volume_ratio = order_size / avg_volume
        
        price_impact = self._estimate_price_impact(volume_ratio)
        volume_impact = self._estimate_volume_impact(volume_ratio)
        spread_impact = self._estimate_spread_impact(market_data)
        
        return ImpactEstimate(
            price_impact=price_impact,
            volume_impact=volume_impact,
            spread_impact=spread_impact,
            total_impact=price_impact + volume_impact + spread_impact
        )
