from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class AnalysisResult:
    success_rate: float
    profit_factor: float
    expectancy: float
    win_loss_ratio: float
    analysis_period: str

class StrategyResultAnalyzer:
    def __init__(self, analysis_config: Dict = None):
        self.config = analysis_config or {
            'min_trades': 20,
            'analysis_period': '1M'
        }
        
    async def analyze_results(self, trade_history: List[Dict]) -> AnalysisResult:
        """전략 결과 분석"""
        if len(trade_history) < self.config['min_trades']:
            return None
            
        trades_df = pd.DataFrame(trade_history)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        
        return AnalysisResult(
            success_rate=len(winning_trades) / len(trades_df),
            profit_factor=self._calculate_profit_factor(trades_df),
            expectancy=self._calculate_expectancy(trades_df),
            win_loss_ratio=self._calculate_win_loss_ratio(trades_df),
            analysis_period=self.config['analysis_period']
        )
