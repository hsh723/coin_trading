import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    volatility: float
    var: float
    cvar: float
    beta: float
    correlation_matrix: pd.DataFrame

class PortfolioRiskMetrics:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
    def calculate_metrics(self, returns: pd.DataFrame, weights: np.ndarray) -> RiskMetrics:
        """포트폴리오 위험 지표 계산"""
        portfolio_returns = self._calculate_portfolio_returns(returns, weights)
        return RiskMetrics(
            volatility=self._calculate_volatility(portfolio_returns),
            var=self._calculate_var(portfolio_returns),
            cvar=self._calculate_cvar(portfolio_returns),
            beta=self._calculate_beta(portfolio_returns, returns),
            correlation_matrix=returns.corr()
        )
