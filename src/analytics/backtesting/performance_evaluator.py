import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float

class PerformanceEvaluator:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def evaluate_strategy(self, trades: pd.DataFrame, equity_curve: pd.Series) -> PerformanceMetrics:
        """전략 성과 평가"""
        returns = equity_curve.pct_change().dropna()
        
        return PerformanceMetrics(
            total_return=self._calculate_total_return(equity_curve),
            annualized_return=self._calculate_annualized_return(returns),
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            win_rate=self._calculate_win_rate(trades),
            profit_factor=self._calculate_profit_factor(trades)
        )
