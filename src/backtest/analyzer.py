import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go

class BacktestAnalyzer:
    def __init__(self, trades: List[Dict], market_data: pd.DataFrame):
        self.trades = pd.DataFrame(trades)
        self.market_data = market_data
        
    def calculate_metrics(self) -> Dict:
        """성과 지표 계산"""
        returns = self.calculate_returns()
        
        return {
            'total_return': returns.sum(),
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor()
        }
        
    def plot_equity_curve(self) -> go.Figure:
        """자본금 곡선 플롯"""
        cumulative_returns = (1 + self.calculate_returns()).cumprod()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name='Equity Curve'
        ))
        return fig
