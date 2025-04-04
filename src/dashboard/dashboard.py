"""
대시보드 모듈
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class Dashboard:
    """대시보드 클래스"""
    
    def __init__(self, trading_bot):
        """
        초기화
        
        Args:
            trading_bot: 트레이딩 봇 객체
        """
        self.bot = trading_bot
        self.market_data = None
        self.positions = []
        self.trades = []
        self.performance = {}
    
    def render(self):
        """대시보드 렌더링"""
        try:
            # 사이드바 설정
            self._render_sidebar()
            
            # 메인 컨텐츠
            st.title("📊 트레이딩 대시보드")
            
            # 실시간 데이터 업데이트
            self._update_data()
            
            # 대시보드 레이아웃
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # 차트 섹션
                st.subheader("📈 시장 차트")
                self._render_market_chart()
                
                # 포지션 섹션
                st.subheader("💰 포지션")
                self._render_positions()
                
                # 거래 내역 섹션
                st.subheader("💱 거래 내역")
                self._render_trades()
            
            with col2:
                # 성과 지표 섹션
                st.subheader("📊 성과 지표")
                self._render_performance_metrics()
                
                # 리스크 지표 섹션
                st.subheader("⚠️ 리스크 지표")
                self._render_risk_metrics()
                
                # 전략 성과 섹션
                st.subheader("🎯 전략 성과")
                self._render_strategy_performance()
            
        except Exception as e:
            logger.error(f"대시보드 렌더링 중 오류 발생: {str(e)}")
            st.error("대시보드 렌더링 중 오류가 발생했습니다.")
    
    def _render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.title("⚙️ 설정")
            
            # 심볼 선택
            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
            selected_symbol = st.selectbox("심볼", symbols)
            
            # 시간대 선택
            timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            selected_timeframe = st.selectbox("시간대", timeframes)
            
            # 전략 설정
            st.subheader("전략 설정")
            strategy_params = {
                'rsi_period': st.slider("RSI 기간", 5, 30, 14),
                'rsi_overbought': st.slider("RSI 과매수", 60, 90, 70),
                'rsi_oversold': st.slider("RSI 과매도", 10, 40, 30),
                'bb_period': st.slider("볼린저 밴드 기간", 10, 50, 20),
                'bb_std': st.slider("볼린저 밴드 표준편차", 1.0, 3.0, 2.0)
            }
            
            # 리스크 설정
            st.subheader("리스크 설정")
            risk_params = {
                'max_position_size': st.slider("최대 포지션 크기 (%)", 1, 50, 10),
                'max_leverage': st.slider("최대 레버리지", 1, 10, 3),
                'max_drawdown': st.slider("최대 낙폭 (%)", 1, 50, 20),
                'risk_per_trade': st.slider("거래당 리스크 (%)", 0.1, 5.0, 1.0)
            }
            
            # 설정 적용 버튼
            if st.button("설정 적용"):
                self.bot.strategy.update_parameters(strategy_params)
                self.bot.risk_manager.update_parameters(risk_params)
                st.success("설정이 적용되었습니다.")
    
    def _update_data(self):
        """데이터 업데이트"""
        try:
            # 시장 데이터 업데이트
            self.market_data = self.bot.get_market_data()
            
            # 포지션 업데이트
            self.positions = self.bot.get_positions()
            
            # 거래 내역 업데이트
            self.trades = self.bot.get_trades()
            
            # 성과 지표 업데이트
            self.performance = self.bot.get_portfolio_metrics()
            
        except Exception as e:
            logger.error(f"데이터 업데이트 중 오류 발생: {str(e)}")
    
    def _render_market_chart(self):
        """시장 차트 렌더링"""
        try:
            if self.market_data is None:
                st.warning("시장 데이터가 없습니다.")
                return
            
            # 차트 생성
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # 캔들스틱 차트
            fig.add_trace(
                go.Candlestick(
                    x=self.market_data.index,
                    open=self.market_data['open'],
                    high=self.market_data['high'],
                    low=self.market_data['low'],
                    close=self.market_data['close'],
                    name="OHLC"
                ),
                row=1, col=1
            )
            
            # 볼린저 밴드
            bb = self.bot.strategy.calculate_bollinger_bands(self.market_data)
            fig.add_trace(
                go.Scatter(
                    x=self.market_data.index,
                    y=bb['upper'],
                    name="BB Upper",
                    line=dict(color='gray', width=1)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.market_data.index,
                    y=bb['lower'],
                    name="BB Lower",
                    line=dict(color='gray', width=1)
                ),
                row=1, col=1
            )
            
            # RSI
            rsi = self.bot.strategy.calculate_rsi(self.market_data)
            fig.add_trace(
                go.Scatter(
                    x=self.market_data.index,
                    y=rsi,
                    name="RSI",
                    line=dict(color='purple', width=1)
                ),
                row=2, col=1
            )
            
            # 거래량
            fig.add_trace(
                go.Bar(
                    x=self.market_data.index,
                    y=self.market_data['volume'],
                    name="Volume"
                ),
                row=3, col=1
            )
            
            # 레이아웃 설정
            fig.update_layout(
                height=800,
                title="시장 차트",
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"시장 차트 렌더링 중 오류 발생: {str(e)}")
            st.error("시장 차트를 표시할 수 없습니다.")
    
    def _render_positions(self):
        """포지션 렌더링"""
        try:
            if not self.positions:
                st.info("현재 포지션이 없습니다.")
                return
            
            # 포지션 테이블
            df = pd.DataFrame(self.positions)
            st.dataframe(df, use_container_width=True)
            
            # 포지션 차트
            fig = go.Figure()
            
            for position in self.positions:
                fig.add_trace(
                    go.Scatter(
                        x=[position['entry_time'], datetime.now()],
                        y=[position['entry_price'], position['current_price']],
                        mode='lines+markers',
                        name=f"{position['symbol']} ({position['side']})"
                    )
                )
            
            fig.update_layout(
                title="포지션 가격 추이",
                xaxis_title="시간",
                yaxis_title="가격"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"포지션 렌더링 중 오류 발생: {str(e)}")
            st.error("포지션 정보를 표시할 수 없습니다.")
    
    def _render_trades(self):
        """거래 내역 렌더링"""
        try:
            if not self.trades:
                st.info("거래 내역이 없습니다.")
                return
            
            # 거래 내역 테이블
            df = pd.DataFrame(self.trades)
            st.dataframe(df, use_container_width=True)
            
            # 수익률 차트
            fig = go.Figure()
            
            cumulative_pnl = np.cumsum([trade['pnl'] for trade in self.trades])
            fig.add_trace(
                go.Scatter(
                    x=df['entry_time'],
                    y=cumulative_pnl,
                    mode='lines',
                    name="누적 수익"
                )
            )
            
            fig.update_layout(
                title="누적 수익률",
                xaxis_title="시간",
                yaxis_title="수익"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"거래 내역 렌더링 중 오류 발생: {str(e)}")
            st.error("거래 내역을 표시할 수 없습니다.")
    
    def _render_performance_metrics(self):
        """성과 지표 렌더링"""
        try:
            if not self.performance:
                st.warning("성과 지표가 없습니다.")
                return
            
            # 기본 지표
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("총 자본금", f"${self.performance['total_capital']:,.2f}")
                st.metric("총 수익률", f"{self.performance['total_returns']:.1%}")
                st.metric("샤프 비율", f"{self.performance['sharpe_ratio']:.2f}")
            
            with col2:
                st.metric("최대 낙폭", f"{self.performance['max_drawdown']:.1%}")
                st.metric("변동성", f"{self.performance['volatility']:.1%}")
                st.metric("승률", f"{self.performance['win_rate']:.1%}")
            
            # 전략별 성과
            st.subheader("전략별 성과")
            for name, metrics in self.performance['strategy_metrics'].items():
                with st.expander(name):
                    st.metric("수익률", f"{metrics['returns']:.1%}")
                    st.metric("샤프 비율", f"{metrics['sharpe_ratio']:.2f}")
                    st.metric("최대 낙폭", f"{metrics['max_drawdown']:.1%}")
                    st.metric("승률", f"{metrics['win_rate']:.1%}")
            
        except Exception as e:
            logger.error(f"성과 지표 렌더링 중 오류 발생: {str(e)}")
            st.error("성과 지표를 표시할 수 없습니다.")
    
    def _render_risk_metrics(self):
        """리스크 지표 렌더링"""
        try:
            risk_metrics = self.bot.risk_manager.get_risk_metrics()
            
            if not risk_metrics:
                st.warning("리스크 지표가 없습니다.")
                return
            
            # 리스크 지표
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("레버리지", f"{risk_metrics.leverage:.2f}x")
                st.metric("마진 레벨", f"{risk_metrics.margin_level:.1%}")
                st.metric("포지션 크기", f"{risk_metrics.position_size:.1%}")
            
            with col2:
                st.metric("변동성", f"{risk_metrics.volatility:.1%}")
                st.metric("VaR (95%)", f"{risk_metrics.var_95:.1%}")
                st.metric("Expected Shortfall", f"{risk_metrics.expected_shortfall:.1%}")
            
            # 경고 메시지
            warnings = self.bot.risk_manager.check_risk_limits(risk_metrics)
            if warnings:
                st.warning("⚠️ 리스크 경고")
                for warning in warnings:
                    st.error(warning)
            
        except Exception as e:
            logger.error(f"리스크 지표 렌더링 중 오류 발생: {str(e)}")
            st.error("리스크 지표를 표시할 수 없습니다.")
    
    def _render_strategy_performance(self):
        """전략 성과 렌더링"""
        try:
            if not self.performance:
                st.warning("전략 성과가 없습니다.")
                return
            
            # 전략 가중치
            st.subheader("전략 가중치")
            weights = self.performance['strategy_weights']
            
            fig = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=.3
            )])
            
            fig.update_layout(title="전략 가중치 분포")
            st.plotly_chart(fig, use_container_width=True)
            
            # 전략별 수익률 비교
            st.subheader("전략별 수익률")
            returns = {
                name: metrics['returns']
                for name, metrics in self.performance['strategy_metrics'].items()
            }
            
            fig = go.Figure(data=[go.Bar(
                x=list(returns.keys()),
                y=list(returns.values())
            )])
            
            fig.update_layout(
                title="전략별 수익률 비교",
                xaxis_title="전략",
                yaxis_title="수익률"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            logger.error(f"전략 성과 렌더링 중 오류 발생: {str(e)}")
            st.error("전략 성과를 표시할 수 없습니다.") 