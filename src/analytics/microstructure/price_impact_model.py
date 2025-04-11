import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PriceImpactModel:
    temporary_impact: float
    permanent_impact: float
    decay_factor: float
    impact_coefficients: Dict[str, float]

class MarketImpactModel:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'decay_rate': 0.5,
            'impact_threshold': 0.001
        }
        
    async def estimate_impact(self, order_size: float, market_data: Dict) -> PriceImpactModel:
        """가격 영향도 모델링"""
        temp_impact = self._calculate_temporary_impact(order_size, market_data)
        perm_impact = self._calculate_permanent_impact(order_size, market_data)
        
        return PriceImpactModel(
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            decay_factor=self._calculate_decay_factor(market_data),
            impact_coefficients=self._estimate_impact_coefficients(market_data)
        )
