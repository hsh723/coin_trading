from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class PerformanceMetrics:
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    strategy_fitness: float

class AdaptivePerformanceStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'performance_window': 30,
            'risk_free_rate': 0.02,
            'min_trades': 10
        }
        
    async def analyze_performance(self, trade_history: pd.DataFrame) -> PerformanceMetrics:
        """적응형 성과 분석"""
        returns = self._calculate_returns(trade_history)
        risk_metrics = self._calculate_risk_metrics(returns)
        
        return PerformanceMetrics(
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            sortino_ratio=self._calculate_sortino_ratio(returns),
            win_rate=self._calculate_win_rate(trade_history),
            profit_factor=self._calculate_profit_factor(trade_history),
            strategy_fitness=self._evaluate_strategy_fitness(risk_metrics)
        )
