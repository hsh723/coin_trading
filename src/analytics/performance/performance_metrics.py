from typing import Dict, List
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class TradingMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_profit_per_trade: float
    max_drawdown: float
    recovery_factor: float

class PerformanceMetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_trading_metrics(self, trades: pd.DataFrame) -> TradingMetrics:
        """트레이딩 성과 지표 계산"""
        profitable_trades = trades[trades['pnl'] > 0]
        losing_trades = trades[trades['pnl'] < 0]
        
        total_trades = len(trades)
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        profit_factor = abs(profitable_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf')
        
        return TradingMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_profit_per_trade=trades['pnl'].mean(),
            max_drawdown=self._calculate_max_drawdown(trades),
            recovery_factor=self._calculate_recovery_factor(trades)
        )
