import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PortfolioRiskMetrics:
    volatility: float
    var: float
    cvar: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float

class RiskMetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self, returns: pd.Series, weights: np.ndarray = None) -> PortfolioRiskMetrics:
        """포트폴리오 위험 지표 계산"""
        volatility = self._calculate_volatility(returns)
        var = self._calculate_var(returns)
        
        return PortfolioRiskMetrics(
            volatility=volatility,
            var=var,
            cvar=self._calculate_cvar(returns, var),
            sharpe_ratio=self._calculate_sharpe_ratio(returns, volatility),
            sortino_ratio=self._calculate_sortino_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(returns)
        )
