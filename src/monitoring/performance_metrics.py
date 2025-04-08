import pandas as pd
from typing import Dict, List
import numpy as np

class PerformanceMetrics:
    def __init__(self):
        self.metrics_history = []
        
    def calculate_metrics(self, trades: List[Dict], equity_curve: pd.Series) -> Dict:
        """성과 지표 계산"""
        returns = pd.Series([t['pnl'] for t in trades])
        
        return {
            'total_return': equity_curve[-1] / equity_curve[0] - 1,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'win_rate': len(returns[returns > 0]) / len(returns),
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum())
        }
