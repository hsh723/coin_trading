import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MarketImpactEstimate:
    temporary_impact: float
    permanent_impact: float
    resilience_time: float
    cost_estimate: Dict[str, float]

class MarketImpactEstimator:
    def __init__(self, decay_factor: float = 0.5):
        self.decay_factor = decay_factor
        
    async def estimate_impact(self, order_size: float, market_data: Dict) -> MarketImpactEstimate:
        """시장 충격 추정"""
        temp_impact = self._estimate_temporary_impact(order_size, market_data)
        perm_impact = self._estimate_permanent_impact(order_size, market_data)
        
        return MarketImpactEstimate(
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            resilience_time=self._estimate_resilience_time(temp_impact),
            cost_estimate=self._calculate_impact_costs(temp_impact, perm_impact)
        )
