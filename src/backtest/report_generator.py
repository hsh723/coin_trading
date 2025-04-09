from typing import Dict
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BacktestReport:
    summary_metrics: Dict
    trade_history: pd.DataFrame
    equity_curve: pd.Series
    monthly_returns: pd.DataFrame
    drawdown_analysis: Dict

class BacktestReportGenerator:
    def __init__(self, template_path: str = None):
        self.template_path = template_path
        
    def generate_report(self, backtest_results: Dict) -> BacktestReport:
        """백테스트 결과 보고서 생성"""
        summary = self._generate_summary_metrics(backtest_results)
        trades = self._process_trade_history(backtest_results['trades'])
        equity = self._create_equity_curve(backtest_results)
        
        return BacktestReport(
            summary_metrics=summary,
            trade_history=trades,
            equity_curve=equity,
            monthly_returns=self._calculate_monthly_returns(equity),
            drawdown_analysis=self._analyze_drawdowns(equity)
        )

class ReportGenerator:
    def __init__(self, results: Dict, strategy_name: str):
        self.results = results
        self.strategy_name = strategy_name
        
    def generate_html_report(self) -> str:
        """HTML 형식 리포트 생성"""
        return f"""
        <html>
            <body>
                <h1>백테스트 결과: {self.strategy_name}</h1>
                {self._generate_summary_section()}
                {self._generate_performance_charts()}
                {self._generate_trades_table()}
            </body>
        </html>
        """
