import plotly.graph_objects as go
from typing import Dict, List
import pandas as pd

class BacktestVisualizer:
    def __init__(self, theme: str = 'dark'):
        self.theme = theme
        
    def create_report_figures(self, backtest_results: Dict) -> List[go.Figure]:
        """백테스트 결과 시각화"""
        figures = []
        
        figures.append(self._create_equity_curve(
            backtest_results['equity_curve'],
            backtest_results['trades']
        ))
        
        figures.append(self._create_drawdown_chart(
            backtest_results['drawdowns']
        ))
        
        figures.append(self._create_monthly_returns_heatmap(
            backtest_results['monthly_returns']
        ))
        
        return figures
