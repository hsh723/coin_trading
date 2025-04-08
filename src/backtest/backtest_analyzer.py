"""
백테스트 분석 모듈
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from typing import Dict, List, Optional
import logging
from .backtest_engine import BacktestResult

logger = logging.getLogger(__name__)

class BacktestAnalyzer:
    """백테스트 분석기 클래스"""
    
    def __init__(self, result: BacktestResult):
        """
        초기화
        
        Args:
            result: 백테스트 결과
        """
        self.result = result
        self.logger = logging.getLogger(__name__)
    
    def get_summary(self) -> Dict[str, float]:
        """백테스트 결과 요약"""
        try:
            return {
                '초기 자본금': self.result.initial_capital,
                '최종 자본금': self.result.final_capital,
                '총 수익률': self.result.total_return * 100,
                '샤프 비율': self.result.sharpe_ratio,
                '최대 낙폭': self.result.max_drawdown * 100,
                '승률': self.result.win_rate * 100,
                '수익 요인': self.result.profit_factor,
                '총 거래 수': self.result.total_trades,
                '승리 거래': self.result.winning_trades,
                '패배 거래': self.result.losing_trades,
                '평균 거래': self.result.avg_trade,
                '평균 승리': self.result.avg_win,
                '평균 패배': self.result.avg_loss,
                '변동성': self.result.risk_metrics['volatility'] * 100,
                'VaR(95%)': self.result.risk_metrics['var_95'] * 100,
                '기대 부족': self.result.risk_metrics['expected_shortfall'] * 100,
                '최대 낙폭 기간': self.result.risk_metrics['max_drawdown_duration']
            }
            
        except Exception as e:
            self.logger.error(f"백테스트 결과 요약 생성 중 오류 발생: {str(e)}")
            raise
    
    def plot_equity_curve(self) -> go.Figure:
        """자본금 곡선 시각화"""
        try:
            fig = go.Figure()
            
            # 자본금 곡선
            fig.add_trace(
                go.Scatter(
                    x=self.result.equity_curve.index,
                    y=self.result.equity_curve['equity'],
                    name='자본금',
                    line=dict(color='blue')
                )
            )
            
            # 최대 낙폭 표시
            cummax = self.result.equity_curve['equity'].cummax()
            drawdown = (self.result.equity_curve['equity'] - cummax) / cummax
            max_drawdown_idx = drawdown.idxmin()
            
            fig.add_trace(
                go.Scatter(
                    x=[max_drawdown_idx],
                    y=[self.result.equity_curve.loc[max_drawdown_idx, 'equity']],
                    name='최대 낙폭',
                    mode='markers',
                    marker=dict(color='red', size=10)
                )
            )
            
            fig.update_layout(
                title='자본금 곡선',
                xaxis_title='날짜',
                yaxis_title='자본금',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"자본금 곡선 시각화 중 오류 발생: {str(e)}")
            raise
    
    def plot_drawdown(self) -> go.Figure:
        """낙폭 시각화"""
        try:
            fig = go.Figure()
            
            # 낙폭 계산
            cummax = self.result.equity_curve['equity'].cummax()
            drawdown = (self.result.equity_curve['equity'] - cummax) / cummax
            
            fig.add_trace(
                go.Scatter(
                    x=self.result.equity_curve.index,
                    y=drawdown * 100,
                    name='낙폭',
                    line=dict(color='red')
                )
            )
            
            fig.update_layout(
                title='낙폭',
                xaxis_title='날짜',
                yaxis_title='낙폭 (%)',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"낙폭 시각화 중 오류 발생: {str(e)}")
            raise
    
    def plot_monthly_returns(self) -> go.Figure:
        """월간 수익률 시각화"""
        try:
            fig = go.Figure()
            
            # 월간 수익률
            fig.add_trace(
                go.Bar(
                    x=self.result.monthly_returns.index,
                    y=self.result.monthly_returns * 100,
                    name='월간 수익률',
                    marker_color=np.where(self.result.monthly_returns >= 0, 'green', 'red')
                )
            )
            
            fig.update_layout(
                title='월간 수익률',
                xaxis_title='날짜',
                yaxis_title='수익률 (%)',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"월간 수익률 시각화 중 오류 발생: {str(e)}")
            raise
    
    def plot_trade_analysis(self) -> go.Figure:
        """거래 분석 시각화"""
        try:
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=('거래 수익률 분포', '거래 기간 분포', '거래 크기 분포', '거래 시간대 분포')
            )
            
            # 거래 수익률 분포
            fig.add_trace(
                go.Histogram(
                    x=self.result.trades['pnl'] / self.result.trades['entry_price'] * 100,
                    name='수익률 분포',
                    nbinsx=50
                ),
                row=1,
                col=1
            )
            
            # 거래 기간 분포
            trade_duration = (self.result.trades['exit_time'] - self.result.trades['entry_time']).dt.total_seconds() / 3600
            fig.add_trace(
                go.Histogram(
                    x=trade_duration,
                    name='거래 기간 분포',
                    nbinsx=50
                ),
                row=1,
                col=2
            )
            
            # 거래 크기 분포
            fig.add_trace(
                go.Histogram(
                    x=self.result.trades['size'],
                    name='거래 크기 분포',
                    nbinsx=50
                ),
                row=2,
                col=1
            )
            
            # 거래 시간대 분포
            trade_hour = self.result.trades['entry_time'].dt.hour
            fig.add_trace(
                go.Histogram(
                    x=trade_hour,
                    name='거래 시간대 분포',
                    nbinsx=24
                ),
                row=2,
                col=2
            )
            
            fig.update_layout(
                title='거래 분석',
                showlegend=True,
                height=800
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"거래 분석 시각화 중 오류 발생: {str(e)}")
            raise
    
    def save_results(self, directory: str):
        """백테스트 결과 저장"""
        try:
            # 요약 저장
            summary_df = pd.DataFrame([self.get_summary()])
            summary_df.to_csv(f"{directory}/summary.csv", index=False)
            
            # 자본금 곡선 저장
            self.result.equity_curve.to_csv(f"{directory}/equity_curve.csv")
            
            # 거래 내역 저장
            self.result.trades.to_csv(f"{directory}/trades.csv", index=False)
            
            # 월간 수익률 저장
            self.result.monthly_returns.to_csv(f"{directory}/monthly_returns.csv")
            
            # 일간 수익률 저장
            self.result.daily_returns.to_csv(f"{directory}/daily_returns.csv")
            
        except Exception as e:
            self.logger.error(f"백테스트 결과 저장 중 오류 발생: {str(e)}")
            raise
    
    def plot_all(self) -> Dict[str, go.Figure]:
        """
        모든 차트 플롯
        
        Returns:
            Dict[str, go.Figure]: 차트 딕셔너리
        """
        return {
            'equity_curve': self.plot_equity_curve(),
            'drawdown': self.plot_drawdown(),
            'monthly_returns': self.plot_monthly_returns(),
            'trade_analysis': self.plot_trade_analysis()
        } 