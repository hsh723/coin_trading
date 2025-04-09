from typing import Dict
import numpy as np
import pandas as pd

class BacktestMetricsCalculator:
    def calculate_metrics(self, returns: pd.Series, trades: pd.DataFrame) -> Dict:
        """백테스트 성과 지표 계산"""
        return {
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(returns.cumsum()),
            'win_rate': len(trades[trades['pnl'] > 0]) / len(trades),
            'profit_factor': self._calculate_profit_factor(trades),
            'avg_trade_duration': self._calculate_avg_trade_duration(trades)
        }
