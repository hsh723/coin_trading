from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class HedgeSignal:
    hedge_ratio: float
    rebalance_needed: bool
    estimated_cost: float
    risk_exposure: Dict[str, float]

class DynamicHedgeStrategy:
    def __init__(self, hedge_config: Dict = None):
        self.config = hedge_config or {
            'min_correlation': 0.6,
            'rebalance_threshold': 0.05,
            'hedge_instruments': ['futures', 'options']
        }
        
    async def calculate_hedge_ratio(self, 
                                  portfolio: Dict,
                                  market_data: Dict) -> HedgeSignal:
        """동적 헤지 비율 계산"""
        correlation = self._calculate_correlation(portfolio, market_data)
        optimal_ratio = self._optimize_hedge_ratio(correlation)
        
        return HedgeSignal(
            hedge_ratio=optimal_ratio,
            rebalance_needed=self._check_rebalance_needed(optimal_ratio),
            estimated_cost=self._estimate_hedge_cost(optimal_ratio),
            risk_exposure=self._calculate_risk_exposure(optimal_ratio)
        )
