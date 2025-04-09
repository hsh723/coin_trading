import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TradeAnalysis:
    win_rate: float
    avg_win_loss_ratio: float
    profit_factor: float
    avg_holding_time: float
    best_performing_hour: int

class TradeHistoryAnalyzer:
    def __init__(self):
        self.trade_history = []
        
    def analyze_trades(self, trades: List[Dict]) -> TradeAnalysis:
        """거래 이력 분석"""
        df = pd.DataFrame(trades)
        profitable_trades = df[df['pnl'] > 0]
        losing_trades = df[df['pnl'] < 0]
        
        return TradeAnalysis(
            win_rate=len(profitable_trades) / len(df) if len(df) > 0 else 0,
            avg_win_loss_ratio=self._calculate_win_loss_ratio(profitable_trades, losing_trades),
            profit_factor=self._calculate_profit_factor(profitable_trades, losing_trades),
            avg_holding_time=self._calculate_avg_holding_time(df),
            best_performing_hour=self._find_best_hour(df)
        )
