from typing import Dict, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BacktestResults:
    total_returns: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trade_analysis: Dict[str, float]

class BacktestResultsAnalyzer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def analyze_results(self, 
                       equity_curve: pd.Series, 
                       trades: pd.DataFrame) -> BacktestResults:
        """백테스트 결과 분석"""
        returns = equity_curve.pct_change().dropna()
        
        return BacktestResults(
            total_returns=self._calculate_total_returns(equity_curve),
            sharpe_ratio=self._calculate_sharpe_ratio(returns),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            win_rate=self._calculate_win_rate(trades),
            profit_factor=self._calculate_profit_factor(trades),
            trade_analysis=self._analyze_trades(trades)
        )
