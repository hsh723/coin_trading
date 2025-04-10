from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class PerformanceStats:
    returns: pd.Series
    drawdown: pd.Series
    rolling_sharpe: pd.Series
    trade_efficiency: float

class StrategyPerformanceTracker:
    def __init__(self, window_size: int = 252):  # 1년
        self.window_size = window_size
        self.performance_history = []
        
    async def track_performance(self, 
                              strategy_id: str,
                              trade_data: Dict) -> PerformanceStats:
        """전략 성과 추적"""
        returns = self._calculate_returns(trade_data)
        drawdown = self._calculate_drawdown(returns)
        
        stats = PerformanceStats(
            returns=returns,
            drawdown=drawdown,
            rolling_sharpe=self._calculate_rolling_sharpe(returns),
            trade_efficiency=self._calculate_trade_efficiency(trade_data)
        )
        
        self.performance_history.append({
            'strategy_id': strategy_id,
            'timestamp': pd.Timestamp.now(),
            'stats': stats
        })
        
        return stats
