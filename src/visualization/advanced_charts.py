"""
고급 차트 시각화 모듈
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

class AdvancedCharts:
    """고급 차트 클래스"""
    
    def __init__(self):
        """초기화"""
        self.logger = logging.getLogger(__name__)
        
    def create_multi_timeframe_chart(self,
                                   data: Dict[str, pd.DataFrame],
                                   indicators: Dict[str, pd.DataFrame],
                                   trades: List[Dict],
                                   timeframe: str = "1h") -> go.Figure:
        """
        멀티 타임프레임 차트 생성
        
        Args:
            data (Dict[str, pd.DataFrame]): 타임프레임별 OHLCV 데이터
            indicators (Dict[str, pd.DataFrame]): 기술적 지표 데이터
            trades (List[Dict]): 거래 기록
            timeframe (str): 기본 타임프레임
            
        Returns:
            go.Figure: Plotly Figure 객체
        """
        try:
            # 서브플롯 생성
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                subplot_titles=("가격 차트", "RSI", "MACD", "볼린저 밴드")
            )
            
            # 기본 타임프레임 데이터
            df = data[timeframe]
            
            # 캔들스틱 차트
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='가격'
                ),
                row=1, col=1
            )
            
            # 거래 포인트 표시
            for trade in trades:
                if trade['timeframe'] == timeframe:
                    # 진입 포인트
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['entry_time']],
                            y=[trade['entry_price']],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up' if trade['side'] == 'buy' else 'triangle-down',
                                size=10,
                                color='green' if trade['side'] == 'buy' else 'red'
                            ),
                            name=f"{trade['side']} 진입"
                        ),
                        row=1, col=1
                    )
                    
                    # 청산 포인트
                    if 'exit_time' in trade and 'exit_price' in trade:
                        fig.add_trace(
                            go.Scatter(
                                x=[trade['exit_time']],
                                y=[trade['exit_price']],
                                mode='markers',
                                marker=dict(
                                    symbol='x',
                                    size=10,
                                    color='blue'
                                ),
                                name='청산'
                            ),
                            row=1, col=1
                        )
            
            # RSI
            if 'RSI' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['RSI'],
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                # RSI 과매수/과매도선
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # MACD
            if 'MACD' in indicators and 'MACD_Signal' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['MACD'],
                        name='MACD',
                        line=dict(color='blue')
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['MACD_Signal'],
                        name='Signal',
                        line=dict(color='orange')
                    ),
                    row=3, col=1
                )
            
            # 볼린저 밴드
            if 'BB_Upper' in indicators and 'BB_Lower' in indicators:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['BB_Upper'],
                        name='BB Upper',
                        line=dict(color='gray')
                    ),
                    row=4, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=indicators['BB_Lower'],
                        name='BB Lower',
                        line=dict(color='gray'),
                        fill='tonexty'
                    ),
                    row=4, col=1
                )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f"{timeframe} 차트",
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                template='plotly_dark'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"멀티 타임프레임 차트 생성 실패: {str(e)}")
            return None
            
    def create_performance_chart(self,
                               performance_data: pd.DataFrame,
                               metrics: List[str] = ['cumulative_return', 'drawdown']) -> go.Figure:
        """
        성과 차트 생성
        
        Args:
            performance_data (pd.DataFrame): 성과 데이터
            metrics (List[str]): 표시할 지표 목록
            
        Returns:
            go.Figure: Plotly Figure 객체
        """
        try:
            fig = go.Figure()
            
            for metric in metrics:
                if metric in performance_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=performance_data.index,
                            y=performance_data[metric],
                            name=metric,
                            mode='lines'
                        )
                    )
            
            fig.update_layout(
                title="성과 분석",
                xaxis_title="날짜",
                yaxis_title="값",
                template='plotly_dark',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"성과 차트 생성 실패: {str(e)}")
            return None
            
    def create_heatmap(self,
                      data: pd.DataFrame,
                      x: str,
                      y: str,
                      z: str,
                      title: str) -> go.Figure:
        """
        히트맵 생성
        
        Args:
            data (pd.DataFrame): 데이터
            x (str): x축 컬럼
            y (str): y축 컬럼
            z (str): 값 컬럼
            title (str): 차트 제목
            
        Returns:
            go.Figure: Plotly Figure 객체
        """
        try:
            # 데이터 피벗
            pivot_data = data.pivot(index=y, columns=x, values=z)
            
            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='RdYlGn',
                    zmid=0
                )
            )
            
            fig.update_layout(
                title=title,
                xaxis_title=x,
                yaxis_title=y,
                template='plotly_dark'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"히트맵 생성 실패: {str(e)}")
            return None
            
    def create_pattern_chart(self,
                           data: pd.DataFrame,
                           patterns: List[Dict]) -> go.Figure:
        """
        패턴 차트 생성
        
        Args:
            data (pd.DataFrame): 가격 데이터
            patterns (List[Dict]): 패턴 정보
            
        Returns:
            go.Figure: Plotly Figure 객체
        """
        try:
            fig = go.Figure()
            
            # 캔들스틱 차트
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='가격'
                )
            )
            
            # 패턴 표시
            for pattern in patterns:
                # 패턴 구간 강조
                fig.add_vrect(
                    x0=pattern['start_time'],
                    x1=pattern['end_time'],
                    fillcolor="lightgray",
                    opacity=0.2,
                    line_width=0
                )
                
                # 패턴 이름 표시
                fig.add_annotation(
                    x=pattern['start_time'],
                    y=pattern['high'],
                    text=pattern['name'],
                    showarrow=True,
                    arrowhead=1
                )
            
            fig.update_layout(
                title="패턴 분석",
                xaxis_rangeslider_visible=False,
                template='plotly_dark'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"패턴 차트 생성 실패: {str(e)}")
            return None 