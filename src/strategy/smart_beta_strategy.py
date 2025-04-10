from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class SmartBetaSignal:
    factor_scores: Dict[str, float]
    portfolio_weights: Dict[str, float]
    rebalance_needed: bool
    risk_metrics: Dict[str, float]

class SmartBetaStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'factors': ['momentum', 'volatility', 'size', 'volume'],
            'rebalance_threshold': 0.1,
            'risk_target': 0.15
        }
        
    async def generate_signals(self, market_data: Dict) -> SmartBetaSignal:
        """스마트 베타 신호 생성"""
        factor_scores = self._calculate_factor_scores(market_data)
        weights = self._optimize_portfolio_weights(factor_scores)
        
        return SmartBetaSignal(
            factor_scores=factor_scores,
            portfolio_weights=weights,
            rebalance_needed=self._check_rebalance_needed(weights),
            risk_metrics=self._calculate_risk_metrics(weights, market_data)
        )
