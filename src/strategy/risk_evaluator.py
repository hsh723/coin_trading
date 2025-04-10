from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class RiskAssessment:
    var_95: float
    max_drawdown: float
    sharpe_ratio: float
    leverage_risk: float
    concentration_risk: float

class RiskEvaluator:
    def __init__(self, risk_config: Dict = None):
        self.config = risk_config or {
            'var_confidence': 0.95,
            'risk_free_rate': 0.02,
            'max_leverage': 3.0
        }
        
    async def evaluate_risk(self, portfolio: Dict, market_data: Dict) -> RiskAssessment:
        """포트폴리오 리스크 평가"""
        returns = self._calculate_returns(portfolio, market_data)
        
        return RiskAssessment(
            var_95=self._calculate_var(returns),
            max_drawdown=self._calculate_max_drawdown(returns),
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            leverage_risk=self._evaluate_leverage_risk(portfolio),
            concentration_risk=self._evaluate_concentration_risk(portfolio)
        )
