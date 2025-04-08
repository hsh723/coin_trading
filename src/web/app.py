"""
암호화폐 트레이딩 봇 웹 인터페이스
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path

from src.trading.trader import Trader
from src.data.collector import DataCollector
from src.strategy.momentum import MomentumStrategy
from src.risk.risk_manager import RiskManager
from src.utils.logger import get_logger

# 로거 초기화
logger = get_logger(__name__)

# 세션 상태 초기화
if 'trader' not in st.session_state:
    st.session_state.trader = None
if 'data_collector' not in st.session_state:
    st.session_state.data_collector = None
if 'strategy' not in st.session_state:
    st.session_state.strategy = None
if 'risk_manager' not in st.session_state:
    st.session_state.risk_manager = None

def load_config() -> Dict[str, Any]:
    """설정 파일 로드"""
    config_path = Path('config/config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def save_config(config: Dict[str, Any]):
    """설정 파일 저장"""
    config_path = Path('config/config.yaml')
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

def initialize_components():
    """컴포넌트 초기화"""
    config = load_config()
    
    if not st.session_state.data_collector:
        st.session_state.data_collector = DataCollector(
            api_key=config.get('api_key', ''),
            api_secret=config.get('api_secret', ''),
            symbols=config.get('symbols', ['BTC/USDT']),
            exchange_id=config.get('exchange_id', 'binance')
        )
        
    if not st.session_state.strategy:
        st.session_state.strategy = MomentumStrategy(
            rsi_period=config.get('strategy', {}).get('rsi_period', 14),
            rsi_overbought=config.get('strategy', {}).get('rsi_overbought', 70),
            rsi_oversold=config.get('strategy', {}).get('rsi_oversold', 30),
            macd_fast=config.get('strategy', {}).get('macd_fast', 12),
            macd_slow=config.get('strategy', {}).get('macd_slow', 26),
            macd_signal=config.get('strategy', {}).get('macd_signal', 9)
        )
        
    if not st.session_state.risk_manager:
        st.session_state.risk_manager = RiskManager(
            max_position_size=config.get('risk', {}).get('max_position_size', 0.1),
            stop_loss=config.get('risk', {}).get('stop_loss', 0.02),
            take_profit=config.get('risk', {}).get('take_profit', 0.04),
            trailing_stop=config.get('risk', {}).get('trailing_stop', 0.01),
            max_daily_loss=config.get('risk', {}).get('max_daily_loss', 0.05),
            max_consecutive_losses=config.get('risk', {}).get('max_consecutive_losses', 3)
        )
        
    if not st.session_state.trader:
        st.session_state.trader = Trader(
            api_key=config.get('api_key', ''),
            api_secret=config.get('api_secret', ''),
            strategy=st.session_state.strategy,
            risk_manager=st.session_state.risk_manager,
            symbols=config.get('symbols', ['BTC/USDT']),
            exchange_id=config.get('exchange_id', 'binance')
        )

def show_dashboard():
    """메인 대시보드 표시"""
    st.title("암호화폐 트레이딩 대시보드")
    
    # 실시간 모니터링 섹션
    st.header("실시간 모니터링")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("계좌 잔고")
        if st.session_state.trader:
            balance = st.session_state.trader.exchange.fetch_balance()
            st.metric("총 자산", f"${balance['total']['USDT']:,.2f}")
            st.metric("사용 가능", f"${balance['free']['USDT']:,.2f}")
    
    with col2:
        st.subheader("포지션")
        if st.session_state.trader:
            for symbol, position in st.session_state.trader.positions.items():
                st.metric(
                    symbol,
                    f"{position['amount']:.4f}",
                    f"${position['unrealized_pnl']:,.2f}"
                )
    
    with col3:
        st.subheader("성과")
        if st.session_state.trader:
            perf = st.session_state.trader.performance
            st.metric("총 거래", perf['total_trades'])
            st.metric("승률", f"{(perf['winning_trades']/perf['total_trades']*100):.1f}%")
            st.metric("총 수익", f"${perf['total_pnl']:,.2f}")
    
    # 실시간 차트
    st.header("가격 차트")
    if st.session_state.data_collector:
        data = st.session_state.data_collector.get_realtime_data('BTC/USDT')
        if not data.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close']
            ))
            st.plotly_chart(fig, use_container_width=True)

def run_backtest():
    """백테스팅 인터페이스"""
    st.title("백테스팅")
    
    # 백테스팅 설정
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox(
            "거래 심볼",
            ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        )
        start_date = st.date_input(
            "시작일",
            datetime.now() - timedelta(days=30)
        )
        end_date = st.date_input(
            "종료일",
            datetime.now()
        )
        
    with col2:
        timeframe = st.selectbox(
            "타임프레임",
            ['1m', '5m', '15m', '1h', '4h', '1d']
        )
        initial_capital = st.number_input(
            "초기 자본금",
            value=10000.0,
            step=1000.0
        )
    
    # 전략 파라미터 설정
    st.header("전략 파라미터")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rsi_period = st.slider(
            "RSI 기간",
            min_value=5,
            max_value=30,
            value=14
        )
        rsi_overbought = st.slider(
            "RSI 과매수",
            min_value=60,
            max_value=90,
            value=70
        )
        
    with col2:
        rsi_oversold = st.slider(
            "RSI 과매도",
            min_value=10,
            max_value=40,
            value=30
        )
        macd_fast = st.slider(
            "MACD 빠른 기간",
            min_value=5,
            max_value=20,
            value=12
        )
        
    with col3:
        macd_slow = st.slider(
            "MACD 느린 기간",
            min_value=20,
            max_value=50,
            value=26
        )
        macd_signal = st.slider(
            "MACD 시그널",
            min_value=5,
            max_value=20,
            value=9
        )
    
    # 백테스팅 실행
    if st.button("백테스팅 실행"):
        with st.spinner("백테스팅 실행 중..."):
            # 백테스팅 로직 구현
            pass

def manage_settings():
    """설정 관리 페이지"""
    st.title("설정")
    
    # API 설정
    st.header("API 설정")
    api_key = st.text_input("API 키", type="password")
    api_secret = st.text_input("API 시크릿", type="password")
    
    # 거래 설정
    st.header("거래 설정")
    symbols = st.multiselect(
        "거래 심볼",
        ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
        default=['BTC/USDT']
    )
    exchange_id = st.selectbox(
        "거래소",
        ['binance', 'upbit', 'bithumb']
    )
    
    # 전략 설정
    st.header("전략 설정")
    strategy_type = st.selectbox(
        "전략 유형",
        ['Momentum', 'Mean Reversion', 'Breakout']
    )
    
    # 리스크 관리 설정
    st.header("리스크 관리 설정")
    col1, col2 = st.columns(2)
    
    with col1:
        max_position_size = st.slider(
            "최대 포지션 크기",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01
        )
        stop_loss = st.slider(
            "손절",
            min_value=0.01,
            max_value=0.1,
            value=0.02,
            step=0.01
        )
        
    with col2:
        take_profit = st.slider(
            "익절",
            min_value=0.01,
            max_value=0.2,
            value=0.04,
            step=0.01
        )
        trailing_stop = st.slider(
            "트레일링 스탑",
            min_value=0.01,
            max_value=0.1,
            value=0.01,
            step=0.01
        )
    
    # 설정 저장
    if st.button("설정 저장"):
        config = {
            'api_key': api_key,
            'api_secret': api_secret,
            'symbols': symbols,
            'exchange_id': exchange_id,
            'strategy': {
                'type': strategy_type,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            },
            'risk': {
                'max_position_size': max_position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'trailing_stop': trailing_stop,
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 3
            }
        }
        save_config(config)
        st.success("설정이 저장되었습니다.")

def display_logs():
    """로그 및 거래 내역 표시"""
    st.title("거래 로그")
    
    # 로그 필터
    col1, col2 = st.columns(2)
    
    with col1:
        log_level = st.selectbox(
            "로그 레벨",
            ['INFO', 'WARNING', 'ERROR', 'ALL']
        )
        
    with col2:
        time_range = st.selectbox(
            "시간 범위",
            ['1시간', '24시간', '7일', '30일', '전체']
        )
    
    # 로그 표시
    if st.session_state.trader:
        logs = st.session_state.trader.logger.handlers[0].stream.getvalue()
        st.text_area("로그", logs, height=400)

def analyze_performance():
    """성과 분석 페이지"""
    st.title("성과 분석")
    
    if not st.session_state.trader:
        st.warning("트레이더가 초기화되지 않았습니다.")
        return
        
    perf = st.session_state.trader.performance
    
    # 성과 지표
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 거래", perf['total_trades'])
    with col2:
        st.metric("승률", f"{(perf['winning_trades']/perf['total_trades']*100):.1f}%")
    with col3:
        st.metric("총 수익", f"${perf['total_pnl']:,.2f}")
    with col4:
        st.metric("최대 손실폭", f"${perf['max_drawdown']:,.2f}")
    
    # 수익률 차트
    st.header("수익률 추이")
    # 차트 구현
    
    # 거래 분포
    st.header("거래 분포")
    # 분포 차트 구현
    
    # 리스크 지표
    st.header("리스크 지표")
    # 리스크 지표 구현

def main():
    """메인 함수"""
    # 사이드바
    st.sidebar.title("암호화폐 트레이딩 봇")
    
    page = st.sidebar.radio(
        "페이지 선택",
        ["대시보드", "백테스팅", "설정", "로그", "성과 분석"]
    )
    
    # 컴포넌트 초기화
    initialize_components()
    
    # 페이지 표시
    if page == "대시보드":
        show_dashboard()
    elif page == "백테스팅":
        run_backtest()
    elif page == "설정":
        manage_settings()
    elif page == "로그":
        display_logs()
    elif page == "성과 분석":
        analyze_performance()

if __name__ == "__main__":
    main() 