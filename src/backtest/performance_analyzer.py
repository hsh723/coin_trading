import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    risk_adjusted_return: float

class BacktestPerformanceAnalyzer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def analyze_performance(self, equity_curve: pd.Series, 
                          trades: pd.DataFrame) -> PerformanceMetrics:
        """백테스트 성과 분석"""
        returns = equity_curve.pct_change().dropna()
        
        return PerformanceMetrics(
            total_return=self._calculate_total_return(equity_curve),
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            sortino_ratio=self._calculate_sortino_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            win_rate=self._calculate_win_rate(trades),
            profit_factor=self._calculate_profit_factor(trades),
            risk_adjusted_return=self._calculate_risk_adjusted_return(returns)
        )
