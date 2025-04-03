import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any
from ..utils.logger import setup_logger

class BacktestVisualizer:
    """백테스팅 결과 시각화 클래스"""
    
    def __init__(self):
        """시각화 클래스 초기화"""
        self.logger = setup_logger('visualizer')
        
    def plot_equity_curve(self, results: Dict[str, Any]) -> go.Figure:
        """
        자본금 곡선 시각화
        
        Args:
            results (Dict[str, Any]): 백테스팅 결과
            
        Returns:
            go.Figure: 자본금 곡선 그래프
        """
        try:
            # 자본금 곡선 데이터 준비
            equity_curve = pd.DataFrame(results['equity_curve'])
            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
            
            # 그래프 생성
            fig = go.Figure()
            
            # 자본금 곡선
            fig.add_trace(go.Scatter(
                x=equity_curve['timestamp'],
                y=equity_curve['equity'],
                name='자본금',
                line=dict(color='blue')
            ))
            
            # 거래 포인트 표시
            trades = results['trades']
            for trade in trades:
                # 진입 포인트
                fig.add_trace(go.Scatter(
                    x=[trade['entry_time']],
                    y=[trade['entry_price'] * trade['size']],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green'
                    ),
                    name='진입',
                    showlegend=False
                ))
                
                # 청산 포인트
                if 'exit_time' in trade:
                    fig.add_trace(go.Scatter(
                        x=[trade['exit_time']],
                        y=[trade['exit_price'] * trade['size']],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color='red'
                        ),
                        name='청산',
                        showlegend=False
                    ))
            
            # 레이아웃 설정
            fig.update_layout(
                title='자본금 곡선',
                xaxis_title='날짜',
                yaxis_title='자본금',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"자본금 곡선 시각화 실패: {str(e)}")
            return go.Figure()
            
    def plot_drawdown(self, results: Dict[str, Any]) -> go.Figure:
        """
        낙폭 시각화
        
        Args:
            results (Dict[str, Any]): 백테스팅 결과
            
        Returns:
            go.Figure: 낙폭 그래프
        """
        try:
            # 자본금 곡선 데이터 준비
            equity_curve = pd.DataFrame(results['equity_curve'])
            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
            
            # 낙폭 계산
            equity_curve['peak'] = equity_curve['equity'].cummax()
            equity_curve['drawdown'] = (equity_curve['equity'] - equity_curve['peak']) / equity_curve['peak']
            
            # 그래프 생성
            fig = go.Figure()
            
            # 낙폭 곡선
            fig.add_trace(go.Scatter(
                x=equity_curve['timestamp'],
                y=equity_curve['drawdown'],
                name='낙폭',
                line=dict(color='red'),
                fill='tozeroy'
            ))
            
            # 레이아웃 설정
            fig.update_layout(
                title='낙폭',
                xaxis_title='날짜',
                yaxis_title='낙폭',
                hovermode='x unified',
                yaxis=dict(tickformat='.1%')
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"낙폭 시각화 실패: {str(e)}")
            return go.Figure()
            
    def plot_monthly_returns(self, results: Dict[str, Any]) -> go.Figure:
        """
        월별 수익률 시각화
        
        Args:
            results (Dict[str, Any]): 백테스팅 결과
            
        Returns:
            go.Figure: 월별 수익률 히트맵
        """
        try:
            # 자본금 곡선 데이터 준비
            equity_curve = pd.DataFrame(results['equity_curve'])
            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
            
            # 월별 수익률 계산
            equity_curve['month'] = equity_curve['timestamp'].dt.to_period('M')
            monthly_returns = equity_curve.groupby('month')['equity'].apply(
                lambda x: (x.iloc[-1] / x.iloc[0]) - 1
            ).reset_index()
            
            # 연도와 월 분리
            monthly_returns['year'] = monthly_returns['month'].dt.year
            monthly_returns['month'] = monthly_returns['month'].dt.month
            
            # 피벗 테이블 생성
            pivot_table = monthly_returns.pivot(
                index='year',
                columns='month',
                values='equity'
            )
            
            # 히트맵 생성
            fig = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='RdYlGn',
                zmid=0
            ))
            
            # 레이아웃 설정
            fig.update_layout(
                title='월별 수익률',
                xaxis_title='월',
                yaxis_title='연도',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"월별 수익률 시각화 실패: {str(e)}")
            return go.Figure()
            
    def plot_trade_distribution(self, results: Dict[str, Any]) -> go.Figure:
        """
        거래 분포 시각화
        
        Args:
            results (Dict[str, Any]): 백테스팅 결과
            
        Returns:
            go.Figure: 거래 분포 그래프
        """
        try:
            # 거래 데이터 준비
            trades = pd.DataFrame(results['trades'])
            
            # 수익/손실 분포
            fig = go.Figure()
            
            # 수익 거래
            profits = trades[trades['pnl'] > 0]['pnl']
            if not profits.empty:
                fig.add_trace(go.Histogram(
                    x=profits,
                    name='수익',
                    marker_color='green',
                    opacity=0.7
                ))
            
            # 손실 거래
            losses = trades[trades['pnl'] < 0]['pnl']
            if not losses.empty:
                fig.add_trace(go.Histogram(
                    x=losses,
                    name='손실',
                    marker_color='red',
                    opacity=0.7
                ))
            
            # 레이아웃 설정
            fig.update_layout(
                title='거래 분포',
                xaxis_title='손익',
                yaxis_title='거래 횟수',
                barmode='overlay'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"거래 분포 시각화 실패: {str(e)}")
            return go.Figure()
            
    def plot_correlation(self, results: Dict[str, Any], benchmark: pd.Series) -> go.Figure:
        """
        벤치마크와의 상관관계 시각화
        
        Args:
            results (Dict[str, Any]): 백테스팅 결과
            benchmark (pd.Series): 벤치마크 수익률
            
        Returns:
            go.Figure: 상관관계 그래프
        """
        try:
            # 자본금 곡선 데이터 준비
            equity_curve = pd.DataFrame(results['equity_curve'])
            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
            
            # 일별 수익률 계산
            equity_curve['returns'] = equity_curve['equity'].pct_change()
            
            # 벤치마크와 일자 맞추기
            benchmark = benchmark.reindex(equity_curve['timestamp'])
            
            # 산점도 생성
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=benchmark,
                y=equity_curve['returns'],
                mode='markers',
                name='일별 수익률'
            ))
            
            # 상관계수 계산
            correlation = equity_curve['returns'].corr(benchmark)
            
            # 레이아웃 설정
            fig.update_layout(
                title=f'벤치마크 상관관계 (상관계수: {correlation:.2f})',
                xaxis_title='벤치마크 수익률',
                yaxis_title='전략 수익률'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"상관관계 시각화 실패: {str(e)}")
            return go.Figure()
            
    def plot_summary(self, results: Dict[str, Any], benchmark: pd.Series = None) -> go.Figure:
        """
        백테스팅 결과 요약 시각화
        
        Args:
            results (Dict[str, Any]): 백테스팅 결과
            benchmark (pd.Series): 벤치마크 수익률 (선택사항)
            
        Returns:
            go.Figure: 요약 그래프
        """
        try:
            # 서브플롯 생성
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    '자본금 곡선', '낙폭',
                    '월별 수익률', '거래 분포',
                    '벤치마크 상관관계', '성과 지표'
                )
            )
            
            # 자본금 곡선
            equity_fig = self.plot_equity_curve(results)
            fig.add_trace(equity_fig.data[0], row=1, col=1)
            
            # 낙폭
            drawdown_fig = self.plot_drawdown(results)
            fig.add_trace(drawdown_fig.data[0], row=1, col=2)
            
            # 월별 수익률
            monthly_fig = self.plot_monthly_returns(results)
            fig.add_trace(monthly_fig.data[0], row=2, col=1)
            
            # 거래 분포
            trade_fig = self.plot_trade_distribution(results)
            for trace in trade_fig.data:
                fig.add_trace(trace, row=2, col=2)
            
            # 벤치마크 상관관계
            if benchmark is not None:
                corr_fig = self.plot_correlation(results, benchmark)
                fig.add_trace(corr_fig.data[0], row=3, col=1)
            
            # 성과 지표 테이블
            metrics = results['metrics']
            metrics_text = '<br>'.join([
                f'{k}: {v:.2%}' if isinstance(v, float) else f'{k}: {v}'
                for k, v in metrics.items()
            ])
            
            fig.add_trace(
                go.Table(
                    header=dict(values=['지표', '값']),
                    cells=dict(values=[
                        list(metrics.keys()),
                        [f'{v:.2%}' if isinstance(v, float) else str(v) for v in metrics.values()]
                    ])
                ),
                row=3, col=2
            )
            
            # 레이아웃 설정
            fig.update_layout(
                height=1200,
                width=1200,
                title_text='백테스팅 결과 요약',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"요약 시각화 실패: {str(e)}")
            return go.Figure() 