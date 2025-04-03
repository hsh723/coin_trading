"""
백테스트 결과 분석 모듈
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger()

class BacktestAnalyzer:
    def __init__(
        self,
        config: Dict[str, Any],
        results: Dict[str, Any]
    ):
        """
        백테스트 분석기 초기화
        
        Args:
            config (Dict[str, Any]): 설정 정보
            results (Dict[str, Any]): 백테스트 결과
        """
        self.config = config
        self.results = results
        self.logger = setup_logger()
        
        # 결과 저장 경로
        self.results_dir = os.path.join('data', 'backtest_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def analyze_results(self):
        """
        백테스트 결과 분석
        """
        try:
            # 성능 지표 계산
            metrics = self._calculate_performance_metrics()
            
            # 시각화 생성
            self._generate_visualizations()
            
            # 상세 리포트 생성
            self._generate_detailed_report(metrics)
            
            self.logger.info("백테스트 결과 분석 완료")
            
        except Exception as e:
            self.logger.error(f"백테스트 결과 분석 실패: {str(e)}")
            raise
            
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """
        성능 지표 계산
        
        Returns:
            Dict[str, float]: 성능 지표
        """
        try:
            # 기본 지표
            total_trades = len(self.results['trades'])
            winning_trades = len([t for t in self.results['trades'] if t['pnl'] > 0])
            losing_trades = total_trades - winning_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 수익률 지표
            total_return = (self.results['final_capital'] - self.results['initial_capital']) / self.results['initial_capital']
            daily_returns = pd.DataFrame(self.results['equity_curve']).set_index('timestamp')['equity'].pct_change()
            avg_return = daily_returns.mean() if not daily_returns.empty else 0
            return_std = daily_returns.std() if not daily_returns.empty else 0
            
            # 리스크 지표
            cumulative_returns = (1 + daily_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min()) if not drawdowns.empty else 0
            
            # 거래 지표
            trade_returns = [t['pnl'] for t in self.results['trades']]
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0
            max_trade_return = max(trade_returns) if trade_returns else 0
            min_trade_return = min(trade_returns) if trade_returns else 0
            profit_factor = abs(sum(r for r in trade_returns if r > 0) / sum(r for r in trade_returns if r < 0)) if any(r < 0 for r in trade_returns) else float('inf')
            
            # 시간 지표
            holding_times = [
                (t['exit_time'] - t['entry_time']).total_seconds() / 3600
                for t in self.results['trades']
            ]
            avg_holding_time = np.mean(holding_times) if holding_times else 0
            max_holding_time = max(holding_times) if holding_times else 0
            
            # 리스크 조정 수익률
            risk_free_rate = 0.02 / 252  # 연간 2% 가정
            sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std() if not daily_returns.empty else 0
            sortino_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns[daily_returns < 0].std() if not daily_returns.empty else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_return': total_return,
                'avg_return': avg_return,
                'return_std': return_std,
                'max_drawdown': max_drawdown,
                'avg_trade_return': avg_trade_return,
                'max_trade_return': max_trade_return,
                'min_trade_return': min_trade_return,
                'profit_factor': profit_factor,
                'avg_holding_time': avg_holding_time,
                'max_holding_time': max_holding_time,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            }
            
        except Exception as e:
            self.logger.error(f"성능 지표 계산 실패: {str(e)}")
            raise
            
    def _generate_visualizations(self):
        """
        시각화 생성
        """
        try:
            # 자본금 곡선
            self._plot_equity_curve()
            
            # 낙폭 분석
            self._plot_drawdown()
            
            # 거래 분포
            self._plot_trade_distribution()
            
            # 월별 수익률
            self._plot_monthly_returns()
            
            # 시간대별 수익률
            self._plot_hourly_returns()
            
            # 요일별 수익률
            self._plot_weekly_returns()
            
        except Exception as e:
            self.logger.error(f"시각화 생성 실패: {str(e)}")
            raise
            
    def _plot_equity_curve(self):
        """
        자본금 곡선 시각화
        """
        try:
            equity_df = pd.DataFrame(self.results['equity_curve'])
            
            fig = go.Figure()
            
            # 자본금 곡선
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['equity'],
                    name='자본금',
                    line=dict(color='blue')
                )
            )
            
            # 거래 포인트
            for trade in self.results['trades']:
                # 진입 포인트
                fig.add_trace(
                    go.Scatter(
                        x=[trade['entry_time']],
                        y=[trade['entry_price']],
                        mode='markers',
                        name=f"진입 ({trade['side']})",
                        marker=dict(
                            color='green' if trade['side'] == 'buy' else 'red',
                            size=10
                        )
                    )
                )
                
                # 청산 포인트
                fig.add_trace(
                    go.Scatter(
                        x=[trade['exit_time']],
                        y=[trade['exit_price']],
                        mode='markers',
                        name=f"청산 ({trade['pnl']:.2f})",
                        marker=dict(
                            color='blue',
                            size=10
                        )
                    )
                )
                
            fig.update_layout(
                title='자본금 곡선 및 거래 포인트',
                xaxis_title='시간',
                yaxis_title='자본금 (USDT)',
                showlegend=True
            )
            
            fig.write_html(
                os.path.join(self.results_dir, 'equity_curve.html')
            )
            
        except Exception as e:
            self.logger.error(f"자본금 곡선 시각화 실패: {str(e)}")
            raise
            
    def _plot_drawdown(self):
        """
        낙폭 분석 시각화
        """
        try:
            equity_df = pd.DataFrame(self.results['equity_curve'])
            daily_returns = equity_df.set_index('timestamp')['equity'].pct_change()
            
            # 누적 수익률
            cumulative_returns = (1 + daily_returns).cumprod()
            
            # 낙폭 계산
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            
            fig = go.Figure()
            
            # 낙폭
            fig.add_trace(
                go.Scatter(
                    x=drawdowns.index,
                    y=drawdowns.values,
                    name='낙폭',
                    line=dict(color='red')
                )
            )
            
            fig.update_layout(
                title='낙폭 분석',
                xaxis_title='시간',
                yaxis_title='낙폭 (%)',
                showlegend=True
            )
            
            fig.write_html(
                os.path.join(self.results_dir, 'drawdown.html')
            )
            
        except Exception as e:
            self.logger.error(f"낙폭 분석 시각화 실패: {str(e)}")
            raise
            
    def _plot_trade_distribution(self):
        """
        거래 분포 시각화
        """
        try:
            # 수익률 분포
            returns = [t['pnl'] for t in self.results['trades']]
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name='수익률 분포',
                    nbinsx=50
                )
            )
            
            fig.update_layout(
                title='거래 수익률 분포',
                xaxis_title='수익률 (USDT)',
                yaxis_title='빈도',
                showlegend=True
            )
            
            fig.write_html(
                os.path.join(self.results_dir, 'trade_distribution.html')
            )
            
        except Exception as e:
            self.logger.error(f"거래 분포 시각화 실패: {str(e)}")
            raise
            
    def _plot_monthly_returns(self):
        """
        월별 수익률 시각화
        """
        try:
            equity_df = pd.DataFrame(self.results['equity_curve'])
            monthly_returns = equity_df.set_index('timestamp')['equity'].resample('M').last().pct_change()
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=monthly_returns.index,
                    y=monthly_returns.values,
                    name='월별 수익률'
                )
            )
            
            fig.update_layout(
                title='월별 수익률',
                xaxis_title='월',
                yaxis_title='수익률 (%)',
                showlegend=True
            )
            
            fig.write_html(
                os.path.join(self.results_dir, 'monthly_returns.html')
            )
            
        except Exception as e:
            self.logger.error(f"월별 수익률 시각화 실패: {str(e)}")
            raise
            
    def _plot_hourly_returns(self):
        """
        시간대별 수익률 시각화
        """
        try:
            equity_df = pd.DataFrame(self.results['equity_curve'])
            hourly_returns = equity_df.set_index('timestamp')['equity'].resample('H').last().pct_change()
            
            # 시간대별 평균 수익률
            hourly_avg_returns = hourly_returns.groupby(hourly_returns.index.hour).mean()
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=hourly_avg_returns.index,
                    y=hourly_avg_returns.values,
                    name='시간대별 평균 수익률'
                )
            )
            
            fig.update_layout(
                title='시간대별 평균 수익률',
                xaxis_title='시간',
                yaxis_title='평균 수익률 (%)',
                showlegend=True
            )
            
            fig.write_html(
                os.path.join(self.results_dir, 'hourly_returns.html')
            )
            
        except Exception as e:
            self.logger.error(f"시간대별 수익률 시각화 실패: {str(e)}")
            raise
            
    def _plot_weekly_returns(self):
        """
        요일별 수익률 시각화
        """
        try:
            equity_df = pd.DataFrame(self.results['equity_curve'])
            daily_returns = equity_df.set_index('timestamp')['equity'].pct_change()
            
            # 요일별 평균 수익률
            weekday_avg_returns = daily_returns.groupby(daily_returns.index.dayofweek).mean()
            
            # 요일 이름 매핑
            weekday_names = ['월', '화', '수', '목', '금', '토', '일']
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Bar(
                    x=[weekday_names[i] for i in weekday_avg_returns.index],
                    y=weekday_avg_returns.values,
                    name='요일별 평균 수익률'
                )
            )
            
            fig.update_layout(
                title='요일별 평균 수익률',
                xaxis_title='요일',
                yaxis_title='평균 수익률 (%)',
                showlegend=True
            )
            
            fig.write_html(
                os.path.join(self.results_dir, 'weekly_returns.html')
            )
            
        except Exception as e:
            self.logger.error(f"요일별 수익률 시각화 실패: {str(e)}")
            raise
            
    def _generate_detailed_report(self, metrics: Dict[str, float]):
        """
        상세 리포트 생성
        
        Args:
            metrics (Dict[str, float]): 성능 지표
        """
        try:
            # 리포트 생성
            report = f"""
# 백테스트 상세 리포트
생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 거래 통계
- 총 거래 수: {metrics['total_trades']}회
- 승리 거래: {metrics['winning_trades']}회
- 패배 거래: {metrics['losing_trades']}회
- 승률: {metrics['win_rate']*100:.2f}%

## 수익률 분석
- 총 수익률: {metrics['total_return']*100:.2f}%
- 일평균 수익률: {metrics['avg_return']*100:.2f}%
- 수익률 표준편차: {metrics['return_std']*100:.2f}%
- 최대 낙폭: {metrics['max_drawdown']*100:.2f}%

## 거래 성과
- 평균 거래 수익률: {metrics['avg_trade_return']:.2f} USDT
- 최대 거래 수익률: {metrics['max_trade_return']:.2f} USDT
- 최소 거래 수익률: {metrics['min_trade_return']:.2f} USDT
- 수익 팩터: {metrics['profit_factor']:.2f}

## 시간 분석
- 평균 보유 시간: {metrics['avg_holding_time']:.2f}시간
- 최대 보유 시간: {metrics['max_holding_time']:.2f}시간

## 리스크 조정 수익률
- 샤프 비율: {metrics['sharpe_ratio']:.2f}
- 소르티노 비율: {metrics['sortino_ratio']:.2f}

## 시각화 결과
- [자본금 곡선](equity_curve.html)
- [낙폭 분석](drawdown.html)
- [거래 분포](trade_distribution.html)
- [월별 수익률](monthly_returns.html)
- [시간대별 수익률](hourly_returns.html)
- [요일별 수익률](weekly_returns.html)
"""
            
            # 리포트 저장
            report_path = os.path.join(
                self.results_dir,
                f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
                
        except Exception as e:
            self.logger.error(f"상세 리포트 생성 실패: {str(e)}")
            raise 