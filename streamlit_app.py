"""
암호화폐 트레이딩 봇 웹 인터페이스
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import yaml
import os
import sys
import threading
import time
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np

# 페이지 설정은 반드시 다른 Streamlit 명령어보다 먼저 와야 함
st.set_page_config(
    page_title="코인 자동매매 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
try:
    from src.bot.trading_bot import TradingBot
    from src.utils.logger import setup_logger
    from src.analysis.performance_analyzer import PerformanceAnalyzer
    from src.database.database import Database
    from src.utils.database import DatabaseManager
    from src.utils.auth import AuthManager
    from src.exchange.binance_exchange import BinanceExchange
    from src.strategy.integrated_strategy import IntegratedStrategy
    from src.risk.risk_manager import RiskManager
    from src.utils.telegram import TelegramNotifier
except ImportError as e:
    st.error(f"모듈 임포트 오류: {str(e)}")

# 환경 변수 로드
load_dotenv()

# CSS 스타일 정의
st.markdown("""
    <style>
    /* 모바일 친화적인 스타일 */
    @media (max-width: 768px) {
        .stButton > button {
            width: 100%;
            margin: 5px 0;
            padding: 12px;
            font-size: 16px;
        }
        .stTextInput > div > div > input {
            width: 100%;
            padding: 12px;
            font-size: 16px;
        }
        .stSelectbox > div > div > div {
            width: 100%;
            padding: 12px;
            font-size: 16px;
        }
        .stNumberInput > div > div > input {
            width: 100%;
            padding: 12px;
            font-size: 16px;
        }
        .stSlider > div > div > div {
            width: 100%;
            padding: 12px;
        }
        .element-container {
            margin-bottom: 1rem;
        }
        .stMarkdown {
            font-size: 16px;
        }
    }
    
    /* 공통 스타일 */
    .main {
        padding: 1rem;
    }
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .chart-container {
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# 로거 설정
logger = setup_logger('streamlit_app')

def init_session_state():
    """세션 상태 초기화"""
    if 'bot' not in st.session_state:
        st.session_state.bot = None
    if 'market_data' not in st.session_state:
        st.session_state.market_data = None
    if 'positions' not in st.session_state:
        st.session_state.positions = []
    if 'trades' not in st.session_state:
        st.session_state.trades = []
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None
    if 'performance_report' not in st.session_state:
        st.session_state.performance_report = None
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv('BINANCE_API_KEY', '')
    if 'api_secret' not in st.session_state:
        st.session_state.api_secret = os.getenv('BINANCE_API_SECRET', '')

def add_log(message: str, level: str = "INFO"):
    """로그 추가"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}"
    
    # 세션 상태 초기화 확인
    if "logs" not in st.session_state:
        st.session_state.logs = []
    
    st.session_state.logs.append(log_entry)
    
    # 로그 파일에도 기록
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)
    
    # 텔레그램으로 알림 전송
    if level in ["ERROR", "WARNING"]:
        telegram.send_message(log_entry)

def load_trading_config():
    """설정 파일 로드"""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def save_trading_config(config):
    """설정 파일 저장"""
    config_path = Path("config/config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

def get_sample_market_data():
    """샘플 시장 데이터 생성"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    return pd.DataFrame({
        'timestamp': date_range,
        'open': [100 + i for i in range(len(date_range))],
        'high': [105 + i for i in range(len(date_range))],
        'low': [95 + i for i in range(len(date_range))],
        'close': [102 + i for i in range(len(date_range))],
        'volume': [1000 + i * 100 for i in range(len(date_range))]
    })

def render_chart(data, symbol: str):
    """차트 렌더링"""
    # 데이터 유효성 검사
    if data is None or (isinstance(data, dict) and 'ohlcv' not in data):
        st.warning("시장 데이터가 없습니다.")
        return None
    
    # 데이터프레임 직접 받는 경우
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = data['ohlcv']
    
    # 추가 검증
    if df.empty:
        st.warning("차트 데이터가 비어 있습니다.")
        return None
        
    # 차트 생성
    fig = go.Figure()
    
    # 캔들스틱 차트
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ))
    
    # 거래량 차트
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        yaxis='y2'
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=f'{symbol} Price Chart',
        xaxis_title='Time',
        yaxis_title='Price',
        yaxis2_title='Volume',
        yaxis2=dict(
            overlaying='y',
            side='right'
        ),
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark'
    )
    
    return fig

def render_performance_metrics(report: dict):
    """성과 지표 렌더링"""
    if not report:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{report['summary']['total_return']:.2%}")
        st.metric("Annual Return", f"{report['summary']['annual_return']:.2%}")
    
    with col2:
        st.metric("Max Drawdown", f"{report['summary']['max_drawdown']:.2%}")
        st.metric("Sharpe Ratio", f"{report['summary']['sharpe_ratio']:.2f}")
    
    with col3:
        st.metric("Win Rate", f"{report['summary']['win_rate']:.2%}")
        st.metric("Total Trades", report['summary']['total_trades'])
    
    with col4:
        st.metric("Profit Factor", f"{report['summary']['profit_factor']:.2f}")
        st.metric("Average Trade Duration", f"{report['trade_analysis']['avg_duration']:.1f} hours")

def render_trade_history(trades: list):
    """거래 내역 렌더링"""
    if not trades:
        return
    
    df = pd.DataFrame(trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
    
    st.dataframe(
        df[['symbol', 'side', 'entry_price', 'exit_price', 'amount', 
            'pnl', 'entry_time', 'exit_time', 'duration']],
        use_container_width=True
    )

def render_position_info(positions: list):
    """포지션 정보 렌더링"""
    if not positions:
        return
    
    df = pd.DataFrame(positions)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['duration'] = (datetime.now() - df['entry_time']).dt.total_seconds() / 3600
    
    st.dataframe(
        df[['symbol', 'entry_price', 'current_price', 'amount', 
            'unrealized_pnl', 'stop_loss', 'take_profit', 'duration']],
        use_container_width=True
    )

# 비동기 함수를 동기식으로 변환하는 유틸리티 함수 추가
def run_async(async_func):
    """비동기 함수를 동기적으로 실행하는 헬퍼 함수"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_func)
        loop.close()
        return result
    except Exception as e:
        print(f"비동기 실행 오류: {e}")
        return None

def create_sample_data():
    """샘플 시장 데이터 생성"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # 랜덤 가격 생성
    np.random.seed(42)
    base_price = 50000
    price_volatility = 0.02
    prices = base_price * (1 + np.random.normal(0, price_volatility, len(date_range)))
    
    # OHLCV 데이터 생성
    df = pd.DataFrame({
        'timestamp': date_range,
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, len(date_range))),
        'low': prices * (1 - np.random.uniform(0, 0.01, len(date_range))),
        'close': prices * (1 + np.random.normal(0, 0.005, len(date_range))),
        'volume': np.random.uniform(100, 1000, len(date_range))
    })
    
    st.session_state.market_data = df
    return df

def update_market_data(exchange, symbol="BTC/USDT", timeframe="1h", limit=100):
    """시장 데이터 업데이트 함수"""
    try:
        # 비동기 함수를 동기적으로 실행
        async def fetch_data():
            return await exchange.fetch_ohlcv(symbol, timeframe, limit)
        
        # 동기식으로 변환하여 실행
        ohlcv_data = run_async(fetch_data())
        
        if ohlcv_data and len(ohlcv_data) > 0:
            # 데이터 가공 및 저장
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            st.session_state.market_data = df
            return True
        else:
            # 샘플 데이터 생성
            create_sample_data()
            return False
    except Exception as e:
        error_msg = f"시장 데이터 업데이트 실패: {str(e)}"
        print(error_msg)  # 로깅
        # 샘플 데이터 생성
        create_sample_data()
        return False

async def update_market_data():
    """시장 데이터 업데이트"""
    try:
        if st.session_state.bot:
            market_data = st.session_state.bot.get_market_data()
            if market_data:
                st.session_state.market_data = market_data
                st.session_state.last_update = datetime.now()
            else:
                create_sample_data()
    except Exception as e:
        st.error(f"시장 데이터 업데이트 실패: {str(e)}")
        create_sample_data()

async def update_positions():
    """포지션 정보 업데이트"""
    try:
        if st.session_state.bot:
            positions = st.session_state.bot.get_positions()
            st.session_state.positions = positions
    except Exception as e:
        st.error(f"포지션 정보 업데이트 실패: {str(e)}")

async def update_trades():
    """거래 내역 업데이트"""
    try:
        if st.session_state.bot:
            trades = st.session_state.bot.get_trades()
            st.session_state.trades = trades
    except Exception as e:
        st.error(f"거래 내역 업데이트 실패: {str(e)}")

async def update_performance_report():
    """성과 리포트 업데이트"""
    try:
        if st.session_state.trades and st.session_state.market_data:
            analyzer = PerformanceAnalyzer()
            report = analyzer.generate_report(
                st.session_state.trades,
                st.session_state.market_data
            )
            st.session_state.performance_report = report
    except Exception as e:
        logger.error(f"성과 리포트 업데이트 실패: {str(e)}")

async def stop_bot():
    """봇 중지"""
    try:
        if st.session_state.bot and st.session_state.bot.is_running:
            await st.session_state.bot.stop()
            st.session_state.bot = None
            st.success("봇이 중지되었습니다.")
    except Exception as e:
        st.error(f"봇 중지 실패: {str(e)}")

def save_api_keys(api_key: str, api_secret: str):
    """API 키를 .env 파일에 저장"""
    env_path = Path('.env')
    
    # 기존 .env 파일 내용 읽기
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    else:
        lines = []
    
    # 기존 키 값 찾기
    key_found = False
    secret_found = False
    new_lines = []
    
    for line in lines:
        if line.startswith('BINANCE_API_KEY='):
            new_lines.append(f'BINANCE_API_KEY={api_key}\n')
            key_found = True
        elif line.startswith('BINANCE_API_SECRET='):
            new_lines.append(f'BINANCE_API_SECRET={api_secret}\n')
            secret_found = True
        else:
            new_lines.append(line)
    
    # 키가 없으면 추가
    if not key_found:
        new_lines.append(f'BINANCE_API_KEY={api_key}\n')
    if not secret_found:
        new_lines.append(f'BINANCE_API_SECRET={api_secret}\n')
    
    # 파일 저장
    with open(env_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    # 환경 변수 업데이트
    os.environ['BINANCE_API_KEY'] = api_key
    os.environ['BINANCE_API_SECRET'] = api_secret
    
    # 세션 상태 업데이트
    st.session_state.api_key = api_key
    st.session_state.api_secret = api_secret

def main():
    """메인 함수"""
    st.title("암호화폐 트레이딩 봇")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("설정")
        
        # API 설정
        api_key = st.text_input("API 키", 
                               value=st.session_state.api_key,
                               type="password")
        api_secret = st.text_input("API 시크릿",
                                  value=st.session_state.api_secret,
                                  type="password")
        
        # API 키가 변경되었을 때만 저장
        if (api_key != st.session_state.api_key or 
            api_secret != st.session_state.api_secret) and api_key and api_secret:
            save_api_keys(api_key, api_secret)
            st.success("API 키가 저장되었습니다.")
        
        # 거래 설정
        symbol = st.selectbox(
            "거래 심볼",
            ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        )
        timeframe = st.selectbox(
            "시간 프레임",
            ["1m", "5m", "15m", "1h", "4h", "1d"]
        )
        initial_capital = st.number_input(
            "초기 자본금",
            min_value=100.0,
            max_value=1000000.0,
            value=10000.0,
            step=100.0
        )
        
        # 봇 제어
        col1, col2 = st.columns(2)
        with col1:
            if st.button("봇 시작"):
                if not st.session_state.bot:
                    config = {
                        'api_key': api_key,
                        'api_secret': api_secret,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'initial_capital': initial_capital,
                        'testnet': True
                    }
                    st.session_state.bot = TradingBot(config)
                    
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        success = loop.run_until_complete(st.session_state.bot.start())
                        if success:
                            st.success("트레이딩 봇이 시작되었습니다.")
                        else:
                            st.error("봇 시작에 실패했습니다.")
                            st.session_state.bot = None
                    except Exception as e:
                        st.error(f"봇 시작 중 오류 발생: {str(e)}")
                        st.session_state.bot = None
                    finally:
                        loop.close()
        
        with col2:
            if st.button("봇 중지"):
                if st.session_state.bot:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(stop_bot())
                    except Exception as e:
                        st.error(f"봇 중지 중 오류 발생: {str(e)}")
                    finally:
                        loop.close()
    
    # 메인 콘텐츠
    tab1, tab2, tab3, tab4 = st.tabs(["차트", "성과", "포지션", "거래 내역"])
    
    with tab1:
        if st.session_state.market_data is not None:
            fig = render_chart(st.session_state.market_data, symbol)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("시장 데이터를 불러오는 중입니다...")
    
    with tab2:
        if st.session_state.performance_report is not None:
            render_performance_metrics(st.session_state.performance_report)
            
            # 성과 차트
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("자본금 곡선")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.market_data.index,
                    y=st.session_state.market_data['equity'],
                    name='Equity'
                ))
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("월별 수익률")
                monthly_returns = st.session_state.performance_report['monthly_analysis']['monthly_stats']
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(monthly_returns.keys()),
                        y=list(monthly_returns.values()),
                        name='Monthly Returns'
                    )
                ])
                fig.update_layout(template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("성과 리포트를 생성하는 중입니다...")
    
    with tab3:
        render_position_info(st.session_state.positions)
    
    with tab4:
        render_trade_history(st.session_state.trades)
    
    # 실시간 업데이트
    if st.session_state.bot and st.session_state.bot.is_running:
        if st.session_state.last_update is None or \
           (datetime.now() - st.session_state.last_update).seconds >= 5:
            try:
                asyncio.run(update_market_data())
                asyncio.run(update_positions())
                asyncio.run(update_trades())
                asyncio.run(update_performance_report())
                st.session_state.last_update = datetime.now()
                st.rerun()
            except Exception as e:
                st.error(f"데이터 업데이트 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    init_session_state()
    main() 