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
import nest_asyncio
from typing import Dict, Any, List, Optional
import numpy as np
from plotly.subplots import make_subplots
from src.bot.trading_bot import TradingBot
from src.utils.database import DatabaseManager
from src.utils.logger import TradeLogger
from src.analysis.technical_analyzer import TechnicalAnalyzer
from src.analysis.self_learning import SelfLearningSystem
from src.strategy.portfolio_manager import PortfolioManager

# 페이지 설정은 반드시 다른 Streamlit 명령어보다 먼저 와야 함
st.set_page_config(
    page_title="코인 자동매매 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 프로젝트 루트 경로를 시스템 경로에 추가
root_path = Path(__file__).parent.absolute()
sys.path.insert(0, str(root_path))

# 지연 임포트 적용
def get_trading_bot():
    from src.bot.trading_bot import TradingBot
    return TradingBot()

def get_exchange():
    from src.exchange.binance_exchange import BinanceExchange
    return BinanceExchange()

def get_database_manager():
    from src.database.database_manager import DatabaseManager
    return DatabaseManager()

def get_logger():
    from src.utils.logger import setup_logger
    return setup_logger('streamlit_app')

def get_technical_analyzer():
    from src.analysis.technical_analyzer import TechnicalAnalyzer
    return TechnicalAnalyzer()

def get_self_learning_system():
    from src.analysis.self_learning import SelfLearningSystem
    return SelfLearningSystem()

# 그 후 모듈 임포트
from src.bot.trading_bot import TradingBot
from src.exchange.binance_exchange import BinanceExchange
from src.database.database_manager import DatabaseManager
from src.utils.monitoring_dashboard import MonitoringDashboard
from src.utils.performance_reporter import PerformanceReporter
from src.utils.feedback_system import FeedbackSystem

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
logger = get_logger()

# 이벤트 루프 관리를 위한 유틸리티
def get_or_create_eventloop():
    """이벤트 루프 가져오기 또는 생성"""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop

# 중첩 이벤트 루프 허용 (Streamlit 환경에서 필요)
nest_asyncio.apply()

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
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

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
    if data is None:
        st.warning("시장 데이터가 없습니다.")
        return None
    
    # 데이터프레임 직접 받는 경우
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict) and 'ohlcv' in data:
        df = data['ohlcv']
    else:
        st.warning("유효한 시장 데이터 형식이 아닙니다.")
        return None
    
    # 데이터프레임 비어있는지 확인
    if df.empty:
        st.warning("차트 데이터가 비어 있습니다.")
        return None
    
    try:
        # 서브플롯 생성
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.1, 
                            row_heights=[0.7, 0.3])
        
        # 캔들스틱 차트 추가
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # 거래량 바 추가
        if 'volume' in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'] if 'timestamp' in df.columns else df.index,
                    y=df['volume'],
                    name='거래량',
                    marker_color='rgba(0, 0, 255, 0.3)'
                ),
                row=2, col=1
            )
        
        # 레이아웃 설정
        fig.update_layout(
            title=f'{symbol} 차트',
            xaxis_title='시간',
            yaxis_title='가격',
            height=600,
            template='plotly_white',
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # X축 레이아웃 설정
        fig.update_xaxes(
            rangeslider_visible=False,
            showgrid=True
        )
        
        # Y축 레이아웃 설정
        fig.update_yaxes(
            showgrid=True,
            row=1, col=1
        )
        
        # 볼륨 Y축 설정
        fig.update_yaxes(
            title_text='거래량',
            showgrid=True,
            row=2, col=1
        )
        
        return fig
    
    except Exception as e:
        st.error(f"차트 렌더링 오류: {str(e)}")
        return None

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

async def start_bot_async():
    """봇 시작 비동기 함수"""
    try:
        if st.session_state.bot:
            await st.session_state.bot.start()
            st.success("봇이 시작되었습니다.")
            return True
        return False
    except Exception as e:
        st.error(f"봇 시작 실패: {str(e)}")
        return False

def start_bot():
    """봇 시작 함수 (동기식 래퍼)"""
    loop = get_or_create_eventloop()
    return loop.run_until_complete(start_bot_async())

async def stop_bot_async():
    """봇 중지 비동기 함수"""
    try:
        if st.session_state.bot and st.session_state.bot.is_running:
            await st.session_state.bot.stop()
            st.session_state.bot = None
            st.success("봇이 중지되었습니다.")
            return True
        return False
    except Exception as e:
        st.error(f"봇 중지 실패: {str(e)}")
        return False

def stop_bot():
    """봇 중지 함수 (동기식 래퍼)"""
    loop = get_or_create_eventloop()
    return loop.run_until_complete(stop_bot_async())

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
                if not api_key or not api_secret:
                    st.error("API 키와 시크릿을 입력해주세요.")
                else:
                    config = {
                        'api_key': api_key,
                        'api_secret': api_secret,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'initial_capital': initial_capital,
                        'testnet': True
                    }
                    st.session_state.bot = TradingBot(config)
                    start_bot()
        
        with col2:
            if st.button("봇 중지"):
                stop_bot()
    
    # 메인 콘텐츠
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["차트", "성과", "포지션", "거래 내역", "로그 및 알림"])
    
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

    # 성과 분석 탭
    with tab2:
        st.title("📊 성과 분석")
        
        # 누적 수익 차트
        st.subheader("📈 누적 수익")
        time_range = st.selectbox("기간 선택", ["일별", "주별", "월별", "전체"])
        
        # 샘플 데이터 생성
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        returns = pd.Series([0.01 * (i % 3 - 1) for i in range(30)], index=dates)
        cumulative_returns = (1 + returns).cumprod()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=cumulative_returns, mode='lines', name='누적 수익'))
        fig.update_layout(title="누적 수익률", xaxis_title="날짜", yaxis_title="수익률")
        st.plotly_chart(fig, use_container_width=True)
        
        # 주요 성과 지표
        st.subheader("📊 주요 성과 지표")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("승률", "65%", "+5%")
        with col2:
            st.metric("손익비", "2.5", "+0.3")
        with col3:
            st.metric("최대 낙폭", "-15%", "-2%")
        with col4:
            st.metric("샤프 비율", "1.8", "+0.2")
        
        # 전략별 성과
        st.subheader("📊 전략별 성과")
        strategies = pd.DataFrame({
            '전략': ['볼린저 밴드', 'RSI', 'MACD', '통합 전략'],
            '수익률': ['+12%', '+8%', '+5%', '+15%'],
            '승률': ['70%', '65%', '60%', '75%'],
            '거래 횟수': [50, 45, 40, 60]
        })
        st.dataframe(strategies, use_container_width=True)
        
        # 시간대별 성과
        st.subheader("📊 시간대별 성과")
        timeframes = pd.DataFrame({
            '시간대': ['아시아', '유럽', '미국'],
            '수익률': ['+8%', '+12%', '+10%'],
            '거래 횟수': [30, 40, 35],
            '평균 수익': ['+0.5%', '+0.8%', '+0.6%']
        })
        st.dataframe(timeframes, use_container_width=True)
        
        # 코인별 성과
        st.subheader("📊 코인별 성과")
        coins = pd.DataFrame({
            '코인': ['BTC', 'ETH', 'SOL', 'BNB'],
            '수익률': ['+15%', '+10%', '+8%', '+12%'],
            '거래 횟수': [25, 20, 15, 18],
            '승률': ['75%', '70%', '65%', '72%']
        })
        st.dataframe(coins, use_container_width=True)

    # 시장 분석 탭
    with tab4:
        st.title("📈 시장 분석")
        
        # 멀티 타임프레임 차트
        st.subheader("📊 멀티 타임프레임 차트")
        selected_timeframe = st.selectbox("시간 프레임 선택", ["5분", "15분", "1시간", "4시간"])
        
        # 샘플 데이터 생성
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5T')
        prices = pd.Series([50000 + i*10 for i in range(100)], index=dates)
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=dates,
            open=prices,
            high=prices + 100,
            low=prices - 100,
            close=prices + 50,
            name='가격'
        ))
        fig.update_layout(title=f"{selected_timeframe} 차트", xaxis_title="시간", yaxis_title="가격")
        st.plotly_chart(fig, use_container_width=True)
        
        # 주요 기술 지표
        st.subheader("📊 주요 기술 지표")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RSI", "45", "-5")
        with col2:
            st.metric("MACD", "0.5", "+0.1")
        with col3:
            st.metric("볼린저 밴드", "중간", "하단")
        
        # 시장 추세 분석
        st.subheader("📈 시장 추세 분석")
        trends = pd.DataFrame({
            '시간 프레임': ['5분', '15분', '1시간', '4시간'],
            '추세': ['하락', '하락', '상승', '상승'],
            '강도': ['강함', '중간', '약함', '중간'],
            '신뢰도': ['높음', '중간', '낮음', '중간']
        })
        st.dataframe(trends, use_container_width=True)
        
        # 뉴스 요약
        st.subheader("📰 뉴스 요약")
        news = pd.DataFrame({
            '시간': ['10분 전', '30분 전', '1시간 전', '2시간 전'],
            '제목': [
                '비트코인, 5만 달러 돌파',
                '이더리움, 런던 하드포크 성공',
                '솔라나, 네트워크 장애 발생',
                '바이낸스, 새로운 상장 코인 발표'
            ],
            '감성': ['긍정', '긍정', '부정', '중립'],
            '영향도': ['높음', '중간', '높음', '낮음']
        })
        st.dataframe(news, use_container_width=True)
        
        # 변동성 분석
        st.subheader("📊 변동성 분석")
        volatility = pd.DataFrame({
            '시간 프레임': ['5분', '15분', '1시간', '4시간'],
            'ATR': ['100', '200', '500', '1000'],
            '변동성': ['높음', '중간', '낮음', '중간'],
            '추세': ['상승', '하락', '상승', '하락']
        })
        st.dataframe(volatility, use_container_width=True)

    # 로그 및 알림 탭
    with tab5:
        st.title("📝 로그 및 알림")
        
        # 필터 설정
        col1, col2, col3 = st.columns(3)
        with col1:
            log_type = st.selectbox("로그 유형", ["전체", "거래", "시스템", "알림"])
        with col2:
            date_range = st.date_input("날짜 범위", [datetime.now() - timedelta(days=7), datetime.now()])
        with col3:
            export_format = st.selectbox("내보내기 형식", ["CSV", "JSON"])
        
        # 로그 표시
        st.subheader("📋 로그 목록")
        
        # 샘플 로그 데이터
        logs = [
            {"timestamp": "2024-01-01 10:00:00", "type": "거래", "message": "BTC/USDT 매수 신호 발생", "level": "INFO"},
            {"timestamp": "2024-01-01 10:01:00", "type": "시스템", "message": "시장 데이터 업데이트 완료", "level": "INFO"},
            {"timestamp": "2024-01-01 10:02:00", "type": "알림", "message": "텔레그램 알림 전송 완료", "level": "INFO"},
            {"timestamp": "2024-01-01 10:03:00", "type": "거래", "message": "ETH/USDT 매도 신호 발생", "level": "INFO"},
            {"timestamp": "2024-01-01 10:04:00", "type": "시스템", "message": "데이터베이스 백업 완료", "level": "INFO"}
        ]
        
        # 로그 필터링
        filtered_logs = logs
        if log_type != "전체":
            filtered_logs = [log for log in logs if log["type"] == log_type]
        
        # 로그 테이블 표시
        log_df = pd.DataFrame(filtered_logs)
        st.dataframe(log_df, use_container_width=True)
        
        # 내보내기 버튼
        if st.button("로그 내보내기"):
            if export_format == "CSV":
                st.error("로그를 CSV로 내보내는 기능은 아직 구현되지 않았습니다.")
            elif export_format == "JSON":
                st.error("로그를 JSON으로 내보내는 기능은 아직 구현되지 않았습니다.")

if __name__ == "__main__":
    init_session_state()
    main() 