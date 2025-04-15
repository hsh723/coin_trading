import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import json
import os
from typing import Dict, List, Optional, Any
import logging

class TradingDashboard:
    """트레이딩 대시보드"""
    
    def __init__(self,
                 data_dir: str = "./data",
                 config_dir: str = "./config"):
        """
        대시보드 초기화
        
        Args:
            data_dir: 데이터 디렉토리
            config_dir: 설정 디렉토리
        """
        self.data_dir = data_dir
        self.config_dir = config_dir
        
        # 로거 설정
        self.logger = logging.getLogger("trading_dashboard")
        
        # 페이지 설정
        st.set_page_config(
            page_title="암호화폐 트레이딩 대시보드",
            page_icon="📊",
            layout="wide"
        )
        
    def run(self) -> None:
        """대시보드 실행"""
        try:
            # 사이드바 설정
            self._setup_sidebar()
            
            # 메인 컨텐츠
            st.title("암호화폐 트레이딩 대시보드")
            
            # 탭 생성
            tab1, tab2, tab3, tab4 = st.tabs([
                "실시간 모니터링",
                "성과 분석",
                "리스크 관리",
                "설정"
            ])
            
            # 실시간 모니터링 탭
            with tab1:
                self._show_realtime_monitoring()
                
            # 성과 분석 탭
            with tab2:
                self._show_performance_analysis()
                
            # 리스크 관리 탭
            with tab3:
                self._show_risk_management()
                
            # 설정 탭
            with tab4:
                self._show_settings()
                
        except Exception as e:
            self.logger.error(f"대시보드 실행 중 오류 발생: {e}")
            st.error(f"오류 발생: {e}")
            
    def _setup_sidebar(self) -> None:
        """사이드바 설정"""
        try:
            with st.sidebar:
                st.header("시스템 상태")
                
                # 시스템 상태 표시
                status = self._get_system_status()
                st.metric("시스템 상태", status["status"])
                st.metric("실행 중인 전략", status["active_strategies"])
                st.metric("활성 포지션", status["active_positions"])
                
                # 자산 정보
                st.header("자산 정보")
                assets = self._get_asset_info()
                for asset, value in assets.items():
                    st.metric(asset, f"${value:,.2f}")
                    
                # 경고 알림
                st.header("경고 알림")
                alerts = self._get_alerts()
                for alert in alerts:
                    st.warning(alert)
                    
        except Exception as e:
            self.logger.error(f"사이드바 설정 중 오류 발생: {e}")
            
    def _show_realtime_monitoring(self) -> None:
        """실시간 모니터링 표시"""
        try:
            # 가격 차트
            st.subheader("가격 차트")
            price_data = self._load_price_data()
            self._plot_price_chart(price_data)
            
            # 포지션 정보
            st.subheader("포지션 정보")
            positions = self._get_position_info()
            st.dataframe(positions)
            
            # 최근 거래
            st.subheader("최근 거래")
            recent_trades = self._get_recent_trades()
            st.dataframe(recent_trades)
            
        except Exception as e:
            self.logger.error(f"실시간 모니터링 표시 중 오류 발생: {e}")
            
    def _show_performance_analysis(self) -> None:
        """성과 분석 표시"""
        try:
            # 성과 지표
            st.subheader("성과 지표")
            metrics = self._get_performance_metrics()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 수익률", f"{metrics['total_return']:.2f}%")
            with col2:
                st.metric("샤프 비율", f"{metrics['sharpe_ratio']:.2f}")
            with col3:
                st.metric("최대 낙폭", f"{metrics['max_drawdown']:.2f}%")
            with col4:
                st.metric("승률", f"{metrics['win_rate']:.2f}%")
                
            # 수익률 차트
            st.subheader("수익률 추이")
            returns_data = self._load_returns_data()
            self._plot_returns_chart(returns_data)
            
            # 거래 분석
            st.subheader("거래 분석")
            trade_analysis = self._get_trade_analysis()
            st.dataframe(trade_analysis)
            
        except Exception as e:
            self.logger.error(f"성과 분석 표시 중 오류 발생: {e}")
            
    def _show_risk_management(self) -> None:
        """리스크 관리 표시"""
        try:
            # 리스크 지표
            st.subheader("리스크 지표")
            risk_metrics = self._get_risk_metrics()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("VaR (95%)", f"${risk_metrics['var_95']:,.2f}")
            with col2:
                st.metric("CVaR (95%)", f"${risk_metrics['cvar_95']:,.2f}")
            with col3:
                st.metric("변동성", f"{risk_metrics['volatility']:.2f}%")
            with col4:
                st.metric("베타", f"{risk_metrics['beta']:.2f}")
                
            # VaR 분석
            st.subheader("VaR 분석")
            var_data = self._load_var_data()
            self._plot_var_chart(var_data)
            
            # 상관관계 분석
            st.subheader("상관관계 분석")
            correlation_data = self._load_correlation_data()
            self._plot_correlation_heatmap(correlation_data)
            
        except Exception as e:
            self.logger.error(f"리스크 관리 표시 중 오류 발생: {e}")
            
    def _show_settings(self) -> None:
        """설정 표시"""
        try:
            # 트레이딩 전략 설정
            st.subheader("트레이딩 전략 설정")
            strategy_config = self._load_strategy_config()
            self._show_strategy_settings(strategy_config)
            
            # 리스크 관리 설정
            st.subheader("리스크 관리 설정")
            risk_config = self._load_risk_config()
            self._show_risk_settings(risk_config)
            
            # 시스템 설정
            st.subheader("시스템 설정")
            system_config = self._load_system_config()
            self._show_system_settings(system_config)
            
        except Exception as e:
            self.logger.error(f"설정 표시 중 오류 발생: {e}")
            
    def _get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        try:
            # 실제 구현에서는 시스템 상태를 모니터링하는 로직 필요
            return {
                "status": "정상",
                "active_strategies": 3,
                "active_positions": 2
            }
        except Exception as e:
            self.logger.error(f"시스템 상태 조회 중 오류 발생: {e}")
            return {
                "status": "오류",
                "active_strategies": 0,
                "active_positions": 0
            }
            
    def _get_asset_info(self) -> Dict[str, float]:
        """자산 정보 조회"""
        try:
            # 실제 구현에서는 자산 정보를 조회하는 로직 필요
            return {
                "총 자산": 100000.0,
                "현금": 50000.0,
                "포지션": 50000.0
            }
        except Exception as e:
            self.logger.error(f"자산 정보 조회 중 오류 발생: {e}")
            return {}
            
    def _get_alerts(self) -> List[str]:
        """경고 알림 조회"""
        try:
            # 실제 구현에서는 경고 알림을 조회하는 로직 필요
            return [
                "변동성 증가 경고",
                "손실 한도 임계치 도달"
            ]
        except Exception as e:
            self.logger.error(f"경고 알림 조회 중 오류 발생: {e}")
            return []
            
    def _load_price_data(self) -> pd.DataFrame:
        """가격 데이터 로드"""
        try:
            # 실제 구현에서는 가격 데이터를 로드하는 로직 필요
            return pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=100),
                'open': np.random.normal(100, 10, 100),
                'high': np.random.normal(105, 10, 100),
                'low': np.random.normal(95, 10, 100),
                'close': np.random.normal(100, 10, 100),
                'volume': np.random.normal(1000, 100, 100)
            })
        except Exception as e:
            self.logger.error(f"가격 데이터 로드 중 오류 발생: {e}")
            return pd.DataFrame()
            
    def _plot_price_chart(self, data: pd.DataFrame) -> None:
        """가격 차트 그리기"""
        try:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                              vertical_spacing=0.03, row_heights=[0.7, 0.3])
            
            # OHLC 차트
            fig.add_trace(go.Candlestick(
                x=data['timestamp'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name='OHLC'
            ), row=1, col=1)
            
            # 거래량 차트
            fig.add_trace(go.Bar(
                x=data['timestamp'],
                y=data['volume'],
                name='Volume'
            ), row=2, col=1)
            
            fig.update_layout(
                title='가격 및 거래량',
                yaxis_title='가격',
                yaxis2_title='거래량',
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"가격 차트 그리기 중 오류 발생: {e}")
            
    def _get_position_info(self) -> pd.DataFrame:
        """포지션 정보 조회"""
        try:
            # 실제 구현에서는 포지션 정보를 조회하는 로직 필요
            return pd.DataFrame({
                'symbol': ['BTC', 'ETH'],
                'position': [1.0, -0.5],
                'entry_price': [50000, 3000],
                'current_price': [51000, 3100],
                'pnl': [1000, -50],
                'pnl_pct': [2.0, -1.67]
            })
        except Exception as e:
            self.logger.error(f"포지션 정보 조회 중 오류 발생: {e}")
            return pd.DataFrame()
            
    def _get_recent_trades(self) -> pd.DataFrame:
        """최근 거래 조회"""
        try:
            # 실제 구현에서는 최근 거래를 조회하는 로직 필요
            return pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=5),
                'symbol': ['BTC', 'ETH', 'BTC', 'ETH', 'BTC'],
                'type': ['LONG', 'SHORT', 'SHORT', 'LONG', 'LONG'],
                'price': [50000, 3000, 51000, 3100, 52000],
                'quantity': [1.0, 0.5, -1.0, 0.5, 1.0],
                'pnl': [1000, -50, 1000, 50, 1000]
            })
        except Exception as e:
            self.logger.error(f"최근 거래 조회 중 오류 발생: {e}")
            return pd.DataFrame()
            
    def _get_performance_metrics(self) -> Dict[str, float]:
        """성과 지표 조회"""
        try:
            # 실제 구현에서는 성과 지표를 조회하는 로직 필요
            return {
                'total_return': 15.5,
                'sharpe_ratio': 1.8,
                'max_drawdown': -5.2,
                'win_rate': 65.0
            }
        except Exception as e:
            self.logger.error(f"성과 지표 조회 중 오류 발생: {e}")
            return {}
            
    def _load_returns_data(self) -> pd.DataFrame:
        """수익률 데이터 로드"""
        try:
            # 실제 구현에서는 수익률 데이터를 로드하는 로직 필요
            return pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=100),
                'returns': np.random.normal(0.001, 0.02, 100)
            })
        except Exception as e:
            self.logger.error(f"수익률 데이터 로드 중 오류 발생: {e}")
            return pd.DataFrame()
            
    def _plot_returns_chart(self, data: pd.DataFrame) -> None:
        """수익률 차트 그리기"""
        try:
            fig = go.Figure()
            
            # 누적 수익률
            cumulative_returns = (1 + data['returns']).cumprod() - 1
            
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=cumulative_returns * 100,
                name='누적 수익률',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title='누적 수익률',
                yaxis_title='수익률 (%)',
                xaxis_title='날짜'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"수익률 차트 그리기 중 오류 발생: {e}")
            
    def _get_trade_analysis(self) -> pd.DataFrame:
        """거래 분석 조회"""
        try:
            # 실제 구현에서는 거래 분석을 조회하는 로직 필요
            return pd.DataFrame({
                'metric': ['평균 수익률', '평균 손실률', '최대 연속 승리', '최대 연속 손실'],
                'value': ['2.5%', '-1.8%', '5', '3']
            })
        except Exception as e:
            self.logger.error(f"거래 분석 조회 중 오류 발생: {e}")
            return pd.DataFrame()
            
    def _get_risk_metrics(self) -> Dict[str, float]:
        """리스크 지표 조회"""
        try:
            # 실제 구현에서는 리스크 지표를 조회하는 로직 필요
            return {
                'var_95': -2500.0,
                'cvar_95': -3000.0,
                'volatility': 2.5,
                'beta': 1.2
            }
        except Exception as e:
            self.logger.error(f"리스크 지표 조회 중 오류 발생: {e}")
            return {}
            
    def _load_var_data(self) -> pd.DataFrame:
        """VaR 데이터 로드"""
        try:
            # 실제 구현에서는 VaR 데이터를 로드하는 로직 필요
            return pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=100),
                'var_95': np.random.normal(-2000, 500, 100),
                'var_99': np.random.normal(-3000, 500, 100)
            })
        except Exception as e:
            self.logger.error(f"VaR 데이터 로드 중 오류 발생: {e}")
            return pd.DataFrame()
            
    def _plot_var_chart(self, data: pd.DataFrame) -> None:
        """VaR 차트 그리기"""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['var_95'],
                name='VaR (95%)',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data['var_99'],
                name='VaR (99%)',
                line=dict(color='orange')
            ))
            
            fig.update_layout(
                title='VaR 추이',
                yaxis_title='VaR ($)',
                xaxis_title='날짜'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"VaR 차트 그리기 중 오류 발생: {e}")
            
    def _load_correlation_data(self) -> pd.DataFrame:
        """상관관계 데이터 로드"""
        try:
            # 실제 구현에서는 상관관계 데이터를 로드하는 로직 필요
            return pd.DataFrame({
                'BTC': [1.0, 0.8, 0.6],
                'ETH': [0.8, 1.0, 0.7],
                'SOL': [0.6, 0.7, 1.0]
            }, index=['BTC', 'ETH', 'SOL'])
        except Exception as e:
            self.logger.error(f"상관관계 데이터 로드 중 오류 발생: {e}")
            return pd.DataFrame()
            
    def _plot_correlation_heatmap(self, data: pd.DataFrame) -> None:
        """상관관계 히트맵 그리기"""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=data.values,
                x=data.columns,
                y=data.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            
            fig.update_layout(
                title='자산 상관관계',
                xaxis_title='자산',
                yaxis_title='자산'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            self.logger.error(f"상관관계 히트맵 그리기 중 오류 발생: {e}")
            
    def _load_strategy_config(self) -> Dict[str, Any]:
        """전략 설정 로드"""
        try:
            # 실제 구현에서는 전략 설정을 로드하는 로직 필요
            return {
                'strategy_type': 'momentum',
                'lookback_period': 20,
                'rsi_threshold': 30,
                'volatility_threshold': 0.02
            }
        except Exception as e:
            self.logger.error(f"전략 설정 로드 중 오류 발생: {e}")
            return {}
            
    def _show_strategy_settings(self, config: Dict[str, Any]) -> None:
        """전략 설정 표시"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.selectbox(
                    "전략 유형",
                    ['momentum', 'mean_reversion', 'breakout'],
                    index=['momentum', 'mean_reversion', 'breakout'].index(config['strategy_type'])
                )
                
                st.number_input(
                    "룩백 기간",
                    min_value=5,
                    max_value=100,
                    value=config['lookback_period']
                )
                
            with col2:
                st.number_input(
                    "RSI 임계값",
                    min_value=0,
                    max_value=100,
                    value=config['rsi_threshold']
                )
                
                st.number_input(
                    "변동성 임계값",
                    min_value=0.0,
                    max_value=0.1,
                    value=config['volatility_threshold'],
                    format="%.4f"
                )
                
        except Exception as e:
            self.logger.error(f"전략 설정 표시 중 오류 발생: {e}")
            
    def _load_risk_config(self) -> Dict[str, Any]:
        """리스크 설정 로드"""
        try:
            # 실제 구현에서는 리스크 설정을 로드하는 로직 필요
            return {
                'max_position_size': 0.2,
                'max_drawdown': 0.1,
                'var_confidence': 0.95,
                'stop_loss': 0.02
            }
        except Exception as e:
            self.logger.error(f"리스크 설정 로드 중 오류 발생: {e}")
            return {}
            
    def _show_risk_settings(self, config: Dict[str, Any]) -> None:
        """리스크 설정 표시"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input(
                    "최대 포지션 크기",
                    min_value=0.0,
                    max_value=1.0,
                    value=config['max_position_size'],
                    format="%.2f"
                )
                
                st.number_input(
                    "최대 낙폭",
                    min_value=0.0,
                    max_value=0.5,
                    value=config['max_drawdown'],
                    format="%.2f"
                )
                
            with col2:
                st.number_input(
                    "VaR 신뢰도",
                    min_value=0.9,
                    max_value=0.99,
                    value=config['var_confidence'],
                    format="%.2f"
                )
                
                st.number_input(
                    "손절매",
                    min_value=0.0,
                    max_value=0.1,
                    value=config['stop_loss'],
                    format="%.2f"
                )
                
        except Exception as e:
            self.logger.error(f"리스크 설정 표시 중 오류 발생: {e}")
            
    def _load_system_config(self) -> Dict[str, Any]:
        """시스템 설정 로드"""
        try:
            # 실제 구현에서는 시스템 설정을 로드하는 로직 필요
            return {
                'data_refresh_interval': 60,
                'log_level': 'INFO',
                'notification_enabled': True
            }
        except Exception as e:
            self.logger.error(f"시스템 설정 로드 중 오류 발생: {e}")
            return {}
            
    def _show_system_settings(self, config: Dict[str, Any]) -> None:
        """시스템 설정 표시"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.number_input(
                    "데이터 갱신 간격 (초)",
                    min_value=1,
                    max_value=3600,
                    value=config['data_refresh_interval']
                )
                
                st.selectbox(
                    "로그 레벨",
                    ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    index=['DEBUG', 'INFO', 'WARNING', 'ERROR'].index(config['log_level'])
                )
                
            with col2:
                st.checkbox(
                    "알림 활성화",
                    value=config['notification_enabled']
                )
                
        except Exception as e:
            self.logger.error(f"시스템 설정 표시 중 오류 발생: {e}") 