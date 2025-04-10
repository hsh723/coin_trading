from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class StrategyPerformance:
    total_return: float
    risk_metrics: Dict[str, float]
    trade_metrics: Dict[str, float]
    drawdown_info: Dict[str, float]

class StrategyPerformanceAnalyzer:
    def __init__(self, metrics_config: Dict = None):
        self.config = metrics_config or {
            'risk_free_rate': 0.02,
            'time_window': '1D'
        }
        
    async def analyze_performance(self, 
                                trades: List[Dict], 
                                market_data: Dict) -> StrategyPerformance:
        """전략 성과 분석"""
        returns = self._calculate_returns(trades)
        
        return StrategyPerformance(
            total_return=self._calculate_total_return(returns),
            risk_metrics=self._calculate_risk_metrics(returns),
            trade_metrics=self._analyze_trades(trades),
            drawdown_info=self._analyze_drawdowns(returns)
        )
