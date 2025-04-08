"""
대시보드 애플리케이션

이 모듈은 트레이딩 시스템의 웹 대시보드를 구현합니다.
주요 기능:
- 실시간 포지션 및 잔고 모니터링
- 거래 내역 및 성과 시각화
- 전략 매개변수 조정
- 알림 설정 관리
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional
import numpy as np
from src.trading.live_simulator import LiveSimulator
from src.trading.strategy import IntegratedStrategy
from src.utils.config_loader import get_config
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
import logging
from src.trading.execution import OrderExecutor
from src.analysis.performance import PerformanceAnalyzer
from src.notification.telegram_bot import TelegramNotifier

# 로거 설정
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="암호화폐 트레이딩 대시보드",
    page_icon="📈",
    layout="wide"
)

# CSS 스타일
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        background-color: #0d6efd;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .position-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }
    .trade-history {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

def create_navigation():
    """네비게이션 바 생성"""
    st.markdown("""
        <div style="background-color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
            <h2 style="margin: 0;">암호화폐 트레이딩 대시보드</h2>
        </div>
    """, unsafe_allow_html=True)

def create_simulation_controls():
    """시뮬레이션 제어 패널"""
    st.sidebar.markdown("### 시뮬레이션 제어")
    
    # 초기 자본금 설정
    initial_capital = st.sidebar.number_input(
        "초기 자본금 (USDT)",
        min_value=1000.0,
        max_value=1000000.0,
        value=10000.0,
        step=1000.0
    )
    
    # 시뮬레이션 속도 설정
    speed = st.sidebar.slider(
        "시뮬레이션 속도",
        min_value=1.0,
        max_value=10.0,
        value=1.0,
        step=0.5
    )
    
    # 리스크 파라미터 설정
    st.sidebar.markdown("### 리스크 관리")
    max_position_size = st.sidebar.slider(
        "최대 포지션 크기 (%)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )
    
    max_loss = st.sidebar.slider(
        "최대 손실 제한 (%)",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=1.0
    )
    
    # 제어 버튼
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("시작"):
            st.session_state.simulation_running = True
            st.session_state.initial_capital = initial_capital
            st.session_state.speed = speed
            st.session_state.max_position_size = max_position_size
            st.session_state.max_loss = max_loss
            
    with col2:
        if st.button("중지"):
            st.session_state.simulation_running = False
            
    with col3:
        if st.button("초기화"):
            st.session_state.simulation_running = False
            st.session_state.positions = {}
            st.session_state.balance = initial_capital
            st.session_state.trade_history = []

def create_portfolio_summary():
    """포트폴리오 요약 정보"""
    if 'positions' not in st.session_state:
        st.session_state.positions = {}
    if 'balance' not in st.session_state:
        st.session_state.balance = st.session_state.initial_capital
        
    total_pnl = sum(pos.get('unrealized_pnl', 0) for pos in st.session_state.positions.values())
    total_value = st.session_state.balance + total_pnl
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 자산", f"{total_value:,.2f} USDT")
        
    with col2:
        st.metric("현금 잔고", f"{st.session_state.balance:,.2f} USDT")
        
    with col3:
        st.metric("미실현 손익", f"{total_pnl:,.2f} USDT")
        
    with col4:
        pnl_percentage = (total_value / st.session_state.initial_capital - 1) * 100
        st.metric("수익률", f"{pnl_percentage:+.2f}%")

def create_positions_view():
    """포지션 정보 표시"""
    st.markdown("### 현재 포지션")
    
    if not st.session_state.positions:
        st.info("현재 보유 중인 포지션이 없습니다.")
        return
        
    for symbol, position in st.session_state.positions.items():
        with st.container():
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"**{symbol}**")
                
            with col2:
                st.markdown(f"방향: {position['side']}")
                
            with col3:
                st.markdown(f"수량: {position['amount']:.4f}")
                
            with col4:
                st.markdown(f"진입가: {position['entry_price']:,.2f}")
                
            with col5:
                pnl = position.get('unrealized_pnl', 0)
                pnl_percentage = (pnl / (position['amount'] * position['entry_price'])) * 100
                color = "green" if pnl >= 0 else "red"
                st.markdown(f"손익: <span style='color: {color}'>{pnl:,.2f} ({pnl_percentage:+.2f}%)</span>", 
                          unsafe_allow_html=True)

def create_trade_history():
    """거래 내역 표시"""
    st.markdown("### 거래 내역")
    
    if 'trade_history' not in st.session_state:
        st.session_state.trade_history = []
        
    if not st.session_state.trade_history:
        st.info("거래 내역이 없습니다.")
        return
        
    trades_df = pd.DataFrame(st.session_state.trade_history)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df = trades_df.sort_values('timestamp', ascending=False)
    
    st.dataframe(
        trades_df.style.format({
            'amount': '{:.4f}',
            'price': '{:,.2f}',
            'fee': '{:.2f}',
            'pnl': '{:,.2f}',
            'balance': '{:,.2f}'
        }),
        use_container_width=True
    )

def create_performance_charts():
    """성과 차트 표시"""
    st.markdown("### 성과 분석")
    
    if 'trade_history' not in st.session_state or not st.session_state.trade_history:
        st.info("성과 데이터가 없습니다.")
        return
        
    trades_df = pd.DataFrame(st.session_state.trade_history)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df = trades_df.sort_values('timestamp')
    
    # 누적 수익률 차트
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=trades_df['timestamp'],
        y=trades_df['balance'],
        mode='lines',
        name='계좌 잔고'
    ))
    fig1.update_layout(
        title='계좌 잔고 추이',
        xaxis_title='시간',
        yaxis_title='잔고 (USDT)',
        height=400
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # 일별 수익률 차트
    daily_returns = trades_df.set_index('timestamp')['pnl'].resample('D').sum()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=daily_returns.index,
        y=daily_returns.values,
        name='일별 수익'
    ))
    fig2.update_layout(
        title='일별 수익률',
        xaxis_title='날짜',
        yaxis_title='수익 (USDT)',
        height=300
    )
    st.plotly_chart(fig2, use_container_width=True)

def main():
    """메인 함수"""
    # 세션 상태 초기화
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    if 'initial_capital' not in st.session_state:
        st.session_state.initial_capital = 10000.0
        
    # 네비게이션 바
    create_navigation()
    
    # 사이드바 컨트롤
    create_simulation_controls()
    
    # 메인 컨텐츠
    if st.session_state.simulation_running:
        # 포트폴리오 요약
        create_portfolio_summary()
        
        # 포지션 정보
        create_positions_view()
        
        # 거래 내역
        create_trade_history()
        
        # 성과 차트
        create_performance_charts()
    else:
        st.info("시뮬레이션을 시작하려면 사이드바의 '시작' 버튼을 클릭하세요.")

class DashboardApp:
    """대시보드 애플리케이션 클래스"""
    
    def __init__(self):
        """대시보드 초기화"""
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.config = get_config()
        
        # 컴포넌트 초기화
        self.executor = None
        self.analyzer = PerformanceAnalyzer(self.config)
        self.notifier = None
        
        # 라우트 등록
        self._register_routes()
        
        # 웹소켓 이벤트 등록
        self._register_socket_events()
        
        logger.info("Dashboard initialized")
    
    def _register_routes(self):
        """라우트 등록"""
        # 메인 페이지
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        # API 엔드포인트
        @self.app.route('/api/positions')
        def get_positions():
            return jsonify(self._get_positions())
        
        @self.app.route('/api/balance')
        def get_balance():
            return jsonify(self._get_balance())
        
        @self.app.route('/api/trades')
        def get_trades():
            return jsonify(self._get_trades())
        
        @self.app.route('/api/performance')
        def get_performance():
            return jsonify(self._get_performance())
        
        @self.app.route('/api/strategy_params', methods=['GET', 'POST'])
        def handle_strategy_params():
            if request.method == 'POST':
                return jsonify(self._update_strategy_params(request.json))
            return jsonify(self._get_strategy_params())
        
        @self.app.route('/api/notifications', methods=['GET', 'POST'])
        def handle_notifications():
            if request.method == 'POST':
                return jsonify(self._update_notification_settings(request.json))
            return jsonify(self._get_notification_settings())
    
    def _register_socket_events(self):
        """웹소켓 이벤트 등록"""
        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("Client disconnected")
    
    def _get_positions(self) -> Dict:
        """현재 포지션 정보 조회"""
        try:
            if not self.executor:
                return {'error': 'Executor not initialized'}
            
            positions = self.executor.get_positions()
            return {
                'positions': positions,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"포지션 정보 조회 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def _get_balance(self) -> Dict:
        """잔고 정보 조회"""
        try:
            if not self.executor:
                return {'error': 'Executor not initialized'}
            
            balance = self.executor.get_balance()
            return {
                'balance': balance,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"잔고 정보 조회 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def _get_trades(self) -> Dict:
        """거래 내역 조회"""
        try:
            trades = self.analyzer.trades
            return {
                'trades': trades,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"거래 내역 조회 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def _get_performance(self) -> Dict:
        """성과 지표 조회"""
        try:
            metrics = self.analyzer.calculate_metrics()
            position_metrics = {
                symbol: self.analyzer.calculate_position_metrics(symbol)
                for symbol in self.analyzer.positions.keys()
            }
            
            return {
                'metrics': metrics.__dict__ if metrics else None,
                'position_metrics': {
                    symbol: metrics.__dict__ if metrics else None
                    for symbol, metrics in position_metrics.items()
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"성과 지표 조회 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def _get_strategy_params(self) -> Dict:
        """전략 매개변수 조회"""
        try:
            return {
                'params': self.config['strategy'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"전략 매개변수 조회 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def _update_strategy_params(self, params: Dict) -> Dict:
        """전략 매개변수 업데이트"""
        try:
            self.config['strategy'].update(params)
            return {
                'success': True,
                'message': '전략 매개변수가 업데이트되었습니다.',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"전략 매개변수 업데이트 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def _get_notification_settings(self) -> Dict:
        """알림 설정 조회"""
        try:
            return {
                'settings': self.config['telegram'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"알림 설정 조회 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def _update_notification_settings(self, settings: Dict) -> Dict:
        """알림 설정 업데이트"""
        try:
            self.config['telegram'].update(settings)
            return {
                'success': True,
                'message': '알림 설정이 업데이트되었습니다.',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"알림 설정 업데이트 중 오류 발생: {str(e)}")
            return {'error': str(e)}
    
    def run(self, host: str = '0.0.0.0', port: int = 5000):
        """
        대시보드 실행
        
        Args:
            host (str): 호스트 주소
            port (int): 포트 번호
        """
        try:
            logger.info(f"Dashboard starting on {host}:{port}")
            self.socketio.run(self.app, host=host, port=port)
            
        except Exception as e:
            logger.error(f"대시보드 실행 중 오류 발생: {str(e)}")
            raise

class Dashboard:
    def __init__(self):
        st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
        
    def run(self):
        """대시보드 실행"""
        st.title("Trading Bot Dashboard")
        
        # 사이드바 메뉴
        menu = st.sidebar.selectbox(
            "메뉴 선택",
            ["실시간 모니터링", "백테스트", "포트폴리오", "설정"]
        )
        
        if menu == "실시간 모니터링":
            self.show_monitoring_page()
        elif menu == "백테스트":
            self.show_backtest_page()
            
    def show_monitoring_page(self):
        """실시간 모니터링 페이지"""
        st.header("실시간 모니터링")
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_price_chart()
        with col2:
            self.show_trading_stats()
            
    def plot_price_chart(self):
        """가격 차트 표시"""
        # 차트 구현...

if __name__ == "__main__":
    main()
    dashboard = Dashboard()
    dashboard.run()