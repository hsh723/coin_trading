from typing import Dict
import pandas as pd
import numpy as np

class BacktestMetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self, equity_curve: pd.Series, trades: pd.DataFrame) -> Dict:
        """백테스트 성과 지표 계산"""
        returns = equity_curve.pct_change().dropna()
        
        return {
            'total_return': self._calculate_total_return(equity_curve),
            'annual_return': self._calculate_annual_return(returns),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(equity_curve),
            'win_rate': self._calculate_win_rate(trades),
            'profit_factor': self._calculate_profit_factor(trades),
            'recovery_factor': self._calculate_recovery_factor(equity_curve),
            'risk_metrics': self._calculate_risk_metrics(returns)
        }
