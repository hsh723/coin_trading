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
    page_title="코인 트레이딩 봇",
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

# 텔레그램 알림 시스템 임포트
from src.notification.telegram_notifier import telegram_notifier

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
    # 기본 상태
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
    
    # API 키 상태
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.getenv('BINANCE_API_KEY', '')
    if 'api_secret' not in st.session_state:
        st.session_state.api_secret = os.getenv('BINANCE_API_SECRET', '')
    
    # 계좌 상태
    if 'account_balance' not in st.session_state:
        st.session_state.account_balance = 0.0
    if 'daily_pnl' not in st.session_state:
        st.session_state.daily_pnl = 0.0
    if 'daily_pnl_pct' not in st.session_state:
        st.session_state.daily_pnl_pct = 0.0
    if 'daily_trades' not in st.session_state:
        st.session_state.daily_trades = 0
    if 'unrealized_pnl' not in st.session_state:
        st.session_state.unrealized_pnl = 0.0
    if 'open_positions' not in st.session_state:
        st.session_state.open_positions = 0
    if 'win_rate' not in st.session_state:
        st.session_state.win_rate = 0.0
    if 'total_trades' not in st.session_state:
        st.session_state.total_trades = 0
    
    # 봇 상태
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    
    # 알림 상태
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'telegram_enabled' not in st.session_state:
        st.session_state.telegram_enabled = False
    if 'notification_types' not in st.session_state:
        st.session_state.notification_types = {
            'trade_signal', 'position_update', 'daily_report', 'error'
        }
    if 'notification_interval' not in st.session_state:
        st.session_state.notification_interval = 5

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

def render_chart(data, symbol: str, indicators: List[str] = None):
    """차트 렌더링"""
    # 데이터 유효성 검사
    if data is None:
        st.warning("시장 데이터가 없습니다.")
        return None
    
    # 데이터프레임 직접 받는 경우
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict) and 'ohlcv' in data:
        df = data['ohlcv'].copy()
    else:
        st.warning("유효한 시장 데이터 형식이 아닙니다.")
        return None
    
    # 데이터프레임 비어있는지 확인
    if df.empty:
        st.warning("차트 데이터가 비어 있습니다.")
        return None
    
    try:
        # 기술적 지표 계산
        if indicators:
            analyzer = TechnicalAnalyzer()
            if 'RSI' in indicators:
                df['RSI'] = analyzer.calculate_rsi(df['close'])
            if 'MACD' in indicators:
                macd_data = analyzer.calculate_macd(df['close'])
                df['MACD'] = macd_data['MACD']
                df['Signal'] = macd_data['Signal']
            if '볼린저밴드' in indicators:
                bb_data = analyzer.calculate_bollinger_bands(df['close'])
                df['BB_Upper'] = bb_data['upper']
                df['BB_Middle'] = bb_data['middle']
                df['BB_Lower'] = bb_data['lower']
            if '이동평균선' in indicators:
                df['MA20'] = analyzer.calculate_ma(df['close'], 20)
                df['MA50'] = analyzer.calculate_ma(df['close'], 50)
                df['MA200'] = analyzer.calculate_ma(df['close'], 200)
        
        # 서브플롯 설정
        subplot_count = 1 + ('RSI' in (indicators or [])) + ('MACD' in (indicators or []))
        heights = [0.5] + [0.25] * (subplot_count - 1)
        fig = make_subplots(rows=subplot_count, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.05, row_heights=heights)
        
        # 캔들스틱 차트 추가
        fig.add_trace(
            go.Candlestick(
                x=df.index if df.index.name == 'timestamp' else df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            ),
            row=1, col=1
        )
        
        # 이동평균선 추가
        if indicators and '이동평균선' in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='orange')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='MA200', line=dict(color='red')), row=1, col=1)
        
        # 볼린저 밴드 추가
        if indicators and '볼린저밴드' in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                                   line=dict(color='gray', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], name='BB Middle',
                                   line=dict(color='gray')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                                   line=dict(color='gray', dash='dash')), row=1, col=1)
        
        # RSI 추가
        if indicators and 'RSI' in indicators:
            current_row = 2
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                                   line=dict(color='purple')), row=current_row, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row)
        
        # MACD 추가
        if indicators and 'MACD' in indicators:
            current_row = 3 if 'RSI' in indicators else 2
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                                   line=dict(color='blue')), row=current_row, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], name='Signal',
                                   line=dict(color='orange')), row=current_row, col=1)
            
            # MACD 히스토그램
            macd_hist = df['MACD'] - df['Signal']
            colors = ['green' if val >= 0 else 'red' for val in macd_hist]
            fig.add_trace(go.Bar(x=df.index, y=macd_hist, name='MACD Histogram',
                               marker_color=colors), row=current_row, col=1)
        
        # 레이아웃 설정
        fig.update_layout(
            title=f'{symbol} 차트',
            xaxis_title='시간',
            yaxis_title='가격',
            height=800,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Y축 레이아웃 설정
        fig.update_yaxes(title_text="가격", row=1, col=1)
        if indicators and 'RSI' in indicators:
            fig.update_yaxes(title_text="RSI", row=2, col=1)
        if indicators and 'MACD' in indicators:
            fig.update_yaxes(title_text="MACD", row=3 if 'RSI' in indicators else 2, col=1)
        
        return fig
    
    except Exception as e:
        logger.error(f"차트 렌더링 오류: {str(e)}")
        return None

def render_performance_metrics(report: dict):
    """성과 지표 렌더링"""
    if not report:
        st.info("성과 데이터가 없습니다.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "총 수익률",
            f"{report['summary']['total_return']:.2%}",
            f"${report['summary']['total_profit']:,.2f}"
        )
        st.metric(
            "연간 수익률",
            f"{report['summary']['annual_return']:.2%}",
            f"월 {report['summary']['monthly_return']:.2%}"
        )
    
    with col2:
        st.metric(
            "최대 낙폭",
            f"{report['summary']['max_drawdown']:.2%}",
            f"{report['summary']['max_drawdown_duration']:.0f}일"
        )
        st.metric(
            "샤프 비율",
            f"{report['summary']['sharpe_ratio']:.2f}",
            f"변동성 {report['summary']['volatility']:.2%}"
        )
    
    with col3:
        st.metric(
            "승률",
            f"{report['summary']['win_rate']:.1%}",
            f"{report['summary']['total_trades']} 거래"
        )
        st.metric(
            "손익비",
            f"{report['summary']['profit_factor']:.2f}",
            f"평균 {report['summary']['avg_trade_return']:.2%}"
        )
    
    with col4:
        st.metric(
            "최대 연속 승리",
            f"{report['summary']['max_win_streak']} 연승",
            f"현재 {report['summary']['current_streak']} 연속"
        )
        st.metric(
            "평균 거래 시간",
            f"{report['summary']['avg_trade_duration']:.1f}시간",
            f"총 {report['summary']['total_trading_days']}일"
        )
    
    # 월별 성과 차트
    st.subheader("📈 월별 성과")
    monthly_returns = pd.DataFrame(report['monthly_analysis']['returns'])
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly_returns.index,
        y=monthly_returns['return'],
        name='월별 수익률',
        marker_color=['red' if x < 0 else 'green' for x in monthly_returns['return']]
    ))
    
    fig.update_layout(
        title='월별 수익률',
        xaxis_title='월',
        yaxis_title='수익률',
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 승률 분포
    st.subheader("📊 승률 분포")
    col1, col2 = st.columns(2)
    
    with col1:
        win_rates = pd.DataFrame(report['trade_analysis']['win_rates'])
        fig = go.Figure(data=[go.Pie(
            labels=win_rates.index,
            values=win_rates['count'],
            hole=.3
        )])
        fig.update_layout(title='승패 비율')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        pnl_dist = pd.DataFrame(report['trade_analysis']['pnl_distribution'])
        fig = go.Figure(data=[go.Bar(
            x=pnl_dist.index,
            y=pnl_dist['count']
        )])
        fig.update_layout(title='손익 분포')
        st.plotly_chart(fig, use_container_width=True)

def render_trade_history(trades: list):
    """거래 내역 렌더링"""
    if not trades:
        return
    
    df = pd.DataFrame(trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['duration'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
    
    # DataFrame을 전치할 때 transpose() 메서드 사용
    display_df = df[['symbol', 'side', 'entry_price', 'exit_price', 'amount', 
                     'pnl', 'entry_time', 'exit_time', 'duration']].copy()
    st.dataframe(display_df, use_container_width=True)

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
        if st.session_state.bot and st.session_state.bot.is_running:
            # 시장 데이터 업데이트
            market_data = await st.session_state.bot.get_market_data()
            if market_data is not None and not market_data.empty:
                st.session_state.market_data = market_data
                st.session_state.last_update = datetime.now()
                
                # 계좌 상태 업데이트
                account_info = await st.session_state.bot.get_account_info()
                if account_info:
                    st.session_state.account_balance = account_info.get('total_balance', 0.0)
                    st.session_state.daily_pnl = account_info.get('daily_pnl', 0.0)
                    st.session_state.daily_pnl_pct = account_info.get('daily_pnl_pct', 0.0)
                    st.session_state.daily_trades = account_info.get('daily_trades', 0)
                
                # 포지션 상태 업데이트
                positions = await st.session_state.bot.get_positions()
                if positions:
                    st.session_state.positions = positions
                    st.session_state.open_positions = len(positions)
                    st.session_state.unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
                
                # 거래 내역 업데이트
                trades = await st.session_state.bot.get_trades()
                if trades:
                    st.session_state.trades = trades
                    st.session_state.total_trades = len(trades)
                    winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
                    st.session_state.win_rate = (winning_trades / len(trades) * 100) if trades else 0
                
                # 성과 리포트 업데이트
                await update_performance_report()
                
                # 텔레그램 알림 전송 (시장 급변 시)
                if market_data is not None and len(market_data) > 1:
                    last_price = market_data['close'].iloc[-1]
                    prev_price = market_data['close'].iloc[-2]
                    price_change = (last_price - prev_price) / prev_price
                    
                    if abs(price_change) >= 0.05:  # 5% 이상 변동
                        direction = "상승" if price_change > 0 else "하락"
                        await telegram_notifier.send_message(
                            f"⚠️ <b>시장 급변</b>\n\n"
                            f"심볼: {st.session_state.bot.symbol}\n"
                            f"가격: ${last_price:,.2f}\n"
                            f"변동: {price_change:.1%} {direction}",
                            "market_alert"
                        )
                
                return True
            else:
                logger.warning("시장 데이터가 비어있습니다. 샘플 데이터를 생성합니다.")
                create_sample_data()
                return False
    except Exception as e:
        logger.error(f"시장 데이터 업데이트 실패: {str(e)}")
        create_sample_data()
        return False

async def update_positions():
    """포지션 정보 업데이트"""
    try:
        if st.session_state.bot and st.session_state.bot.is_running:
            positions = await st.session_state.bot.get_positions()
            if positions is not None:
                st.session_state.positions = positions
                st.session_state.open_positions = len(positions)
                st.session_state.unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
                return True
    except Exception as e:
        logger.error(f"포지션 정보 업데이트 실패: {str(e)}")
    return False

async def update_trades():
    """거래 내역 업데이트"""
    try:
        if st.session_state.bot and st.session_state.bot.is_running:
            trades = await st.session_state.bot.get_trades()
            if trades is not None:
                st.session_state.trades = trades
                st.session_state.total_trades = len(trades)
                winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
                st.session_state.win_rate = (winning_trades / len(trades) * 100) if trades else 0
                return True
    except Exception as e:
        logger.error(f"거래 내역 업데이트 실패: {str(e)}")
    return False

async def update_performance_report():
    """성과 리포트 업데이트"""
    try:
        if st.session_state.trades and st.session_state.market_data is not None:
            # 성과 분석기 초기화
            analyzer = PerformanceAnalyzer(database_manager)
            
            # 일일 리포트 생성
            daily_report = await generate_daily_report()
            
            # 전체 성과 리포트 생성
            report = analyzer.generate_report(
                st.session_state.trades,
                st.session_state.market_data
            )
            
            if report:
                st.session_state.performance_report = report
                
                # 일일 리포트 알림 전송 (매일 00:00)
                now = datetime.now()
                if now.hour == 0 and now.minute == 0:
                    await telegram_notifier.send_daily_report(daily_report)
                
                return True
    except Exception as e:
        logger.error(f"성과 리포트 업데이트 실패: {str(e)}")
        return False

async def generate_daily_report() -> dict:
    """
    일일 성과 리포트 생성
    
    Returns:
        dict: 일일 성과 리포트
    """
    try:
        # 오늘 날짜의 거래만 필터링
        today = datetime.now().date()
        today_trades = [
            t for t in st.session_state.trades
            if pd.to_datetime(t['exit_time']).date() == today
        ]
        
        # 승리/패배 거래
        winning_trades = [t for t in today_trades if t['pnl'] > 0]
        losing_trades = [t for t in today_trades if t['pnl'] < 0]
        
        # 수익/손실
        total_pnl = sum(t['pnl'] for t in today_trades)
        max_profit = max((t['pnl'] for t in today_trades), default=0)
        max_loss = min((t['pnl'] for t in today_trades), default=0)
        
        # 시작/종료 자본금
        start_balance = st.session_state.account_balance - total_pnl
        end_balance = st.session_state.account_balance
        
        report = {
            'date': today.strftime('%Y-%m-%d'),
            'total_trades': len(today_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(today_trades) if today_trades else 0,
            'pnl': total_pnl,
            'return_pct': (end_balance - start_balance) / start_balance if start_balance > 0 else 0,
            'max_profit': max_profit,
            'max_loss': max_loss
        }
        
        return report
        
    except Exception as e:
        logger.error(f"일일 리포트 생성 실패: {str(e)}")
        return {}

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

def setup_telegram():
    """텔레그램 알림 설정"""
    telegram_notifier.setup(
        enabled=st.session_state.telegram_enabled,
        notification_types=st.session_state.notification_types,
        min_interval=st.session_state.notification_interval
    )

async def close_position(symbol: str, amount: float = None):
    """포지션 청산"""
    try:
        if st.session_state.bot and st.session_state.bot.is_running:
            result = await st.session_state.bot.close_position(symbol, amount)
            if result:
                message = f"포지션 청산 성공: {symbol}"
                if amount:
                    message += f" ({amount} 수량)"
                st.success(message)
                
                # 텔레그램 알림 전송
                await telegram_notifier.send_position_update({
                    'symbol': symbol,
                    'status': '청산 완료',
                    'pnl': result.get('pnl', 0),
                    'pnl_pct': result.get('pnl_pct', 0)
                })
                
                return True
            else:
                st.error(f"포지션 청산 실패: {symbol}")
                return False
    except Exception as e:
        st.error(f"포지션 청산 중 오류 발생: {str(e)}")
        return False

async def modify_position(symbol: str, stop_loss: float = None, take_profit: float = None):
    """포지션 수정"""
    try:
        if st.session_state.bot and st.session_state.bot.is_running:
            result = await st.session_state.bot.modify_position(
                symbol, stop_loss=stop_loss, take_profit=take_profit
            )
            if result:
                message = f"포지션 수정 성공: {symbol}"
                if stop_loss:
                    message += f"\n스탑로스: ${stop_loss:,.2f}"
                if take_profit:
                    message += f"\n익절가: ${take_profit:,.2f}"
                st.success(message)
                return True
            else:
                st.error(f"포지션 수정 실패: {symbol}")
                return False
    except Exception as e:
        st.error(f"포지션 수정 중 오류 발생: {str(e)}")
        return False

def filter_trades(trades: list, symbol: str = None, result: str = None, period: str = None) -> list:
    """
    거래 내역 필터링
    
    Args:
        trades (list): 거래 내역 목록
        symbol (str): 심볼 필터
        result (str): 거래 결과 필터 (수익/손실)
        period (str): 기간 필터
        
    Returns:
        list: 필터링된 거래 내역
    """
    if not trades:
        return []
    
    filtered_trades = trades.copy()
    
    # 심볼 필터
    if symbol and symbol != "전체":
        filtered_trades = [t for t in filtered_trades if t['symbol'] == symbol]
    
    # 거래 결과 필터
    if result == "수익":
        filtered_trades = [t for t in filtered_trades if t['pnl'] > 0]
    elif result == "손실":
        filtered_trades = [t for t in filtered_trades if t['pnl'] < 0]
    
    # 기간 필터
    now = datetime.now()
    if period == "오늘":
        filtered_trades = [
            t for t in filtered_trades
            if pd.to_datetime(t['exit_time']).date() == now.date()
        ]
    elif period == "1주일":
        week_ago = now - timedelta(days=7)
        filtered_trades = [
            t for t in filtered_trades
            if pd.to_datetime(t['exit_time']) >= week_ago
        ]
    elif period == "1개월":
        month_ago = now - timedelta(days=30)
        filtered_trades = [
            t for t in filtered_trades
            if pd.to_datetime(t['exit_time']) >= month_ago
        ]
    elif period == "3개월":
        three_months_ago = now - timedelta(days=90)
        filtered_trades = [
            t for t in filtered_trades
            if pd.to_datetime(t['exit_time']) >= three_months_ago
        ]
    
    return filtered_trades

def calculate_trade_stats(trades: list) -> dict:
    """
    거래 통계 계산
    
    Args:
        trades (list): 거래 내역 목록
        
    Returns:
        dict: 거래 통계
    """
    if not trades:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'max_profit': 0,
            'max_loss': 0,
            'profit_factor': 0,
            'total_profit': 0,
            'total_loss': 0
        }
    
    # 기본 통계
    winning_trades = [t for t in trades if t['pnl'] > 0]
    losing_trades = [t for t in trades if t['pnl'] < 0]
    
    total_profit = sum(t['pnl'] for t in winning_trades)
    total_loss = abs(sum(t['pnl'] for t in losing_trades))
    
    stats = {
        'total_trades': len(trades),
        'winning_trades': len(winning_trades),
        'losing_trades': len(losing_trades),
        'win_rate': len(winning_trades) / len(trades) if trades else 0,
        'avg_profit': total_profit / len(winning_trades) if winning_trades else 0,
        'max_profit': max(t['pnl'] for t in trades) if trades else 0,
        'max_loss': min(t['pnl'] for t in trades) if trades else 0,
        'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
        'total_profit': total_profit,
        'total_loss': total_loss
    }
    
    return stats

def main():
    """메인 함수"""
    st.title("암호화폐 트레이딩 봇 🤖")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # API 설정
        with st.expander("API 설정", expanded=False):
            api_key = st.text_input("API 키", 
                                value=st.session_state.api_key,
                                type="password")
            api_secret = st.text_input("API 시크릿",
                                    value=st.session_state.api_secret,
                                    type="password")
            
            if (api_key != st.session_state.api_key or 
                api_secret != st.session_state.api_secret) and api_key and api_secret:
                save_api_keys(api_key, api_secret)
                st.success("✅ API 키가 저장되었습니다.")
        
        # 거래 설정
        with st.expander("거래 설정", expanded=True):
            symbol = st.selectbox(
                "거래 심볼",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
            )
            timeframe = st.selectbox(
                "기본 시간 프레임",
                ["1m", "5m", "15m", "1h", "4h", "1d"]
            )
            initial_capital = st.number_input(
                "초기 자본금 (USDT)",
                min_value=100.0,
                max_value=1000000.0,
                value=10000.0,
                step=100.0
            )
            
            # 리스크 관리 설정
            st.subheader("리스크 관리")
            risk_per_trade = st.slider(
                "거래당 리스크 (%)",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            )
            max_trades = st.number_input(
                "최대 동시 거래 수",
                min_value=1,
                max_value=10,
                value=3
            )
        
        # 알림 설정
        with st.expander("알림 설정", expanded=False):
            telegram_enabled = st.checkbox(
                "텔레그램 알림 활성화",
                value=st.session_state.telegram_enabled
            )
            if telegram_enabled != st.session_state.telegram_enabled:
                st.session_state.telegram_enabled = telegram_enabled
                setup_telegram()
            
            if telegram_enabled:
                bot_token = st.text_input(
                    "텔레그램 봇 토큰",
                    type="password",
                    value=os.getenv('TELEGRAM_BOT_TOKEN', '')
                )
                chat_id = st.text_input(
                    "텔레그램 채팅 ID",
                    value=os.getenv('TELEGRAM_CHAT_ID', '')
                )
                
                if bot_token and chat_id:
                    # .env 파일에 저장
                    with open('.env', 'a') as f:
                        f.write(f"\nTELEGRAM_BOT_TOKEN={bot_token}")
                        f.write(f"\nTELEGRAM_CHAT_ID={chat_id}")
                    os.environ['TELEGRAM_BOT_TOKEN'] = bot_token
                    os.environ['TELEGRAM_CHAT_ID'] = chat_id
                
                notification_types = st.multiselect(
                    "알림 설정",
                    ["진입 신호", "청산 신호", "손절", "익절", "시장 급변", "일일 리포트"],
                    default=list(st.session_state.notification_types)
                )
                st.session_state.notification_types = set(notification_types)
                
                notification_interval = st.slider(
                    "최소 알림 간격 (분)",
                    0, 60, st.session_state.notification_interval
                )
                if notification_interval != st.session_state.notification_interval:
                    st.session_state.notification_interval = notification_interval
                    setup_telegram()
        
        # 봇 제어
        st.header("🎮 봇 제어")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("봇 시작", use_container_width=True):
                if not api_key or not api_secret:
                    st.error("❌ API 키와 시크릿을 입력해주세요.")
                else:
                    config = {
                        'api_key': api_key,
                        'api_secret': api_secret,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'initial_capital': initial_capital,
                        'risk_per_trade': risk_per_trade,
                        'max_trades': max_trades,
                        'testnet': True
                    }
                    st.session_state.bot = TradingBot(config)
                    start_bot()
        
        with col2:
            if st.button("봇 중지", use_container_width=True):
                stop_bot()
    
    # 메인 콘텐츠
    tabs = st.tabs(["📊 대시보드", "📈 차트", "💰 성과", "📋 포지션", "📝 거래 내역", "🔔 알림"])
    
    # 대시보드 탭
    with tabs[0]:
        st.header("📊 대시보드")
        
        # 계좌 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "계좌 잔고",
                f"${st.session_state.get('account_balance', 0):,.2f}",
                f"{st.session_state.get('daily_pnl_pct', 0):.2f}%"
            )
        with col2:
            st.metric(
                "당일 손익",
                f"${st.session_state.get('daily_pnl', 0):,.2f}",
                f"{st.session_state.get('daily_trades', 0)} 거래"
            )
        with col3:
            st.metric(
                "미실현 손익",
                f"${st.session_state.get('unrealized_pnl', 0):,.2f}",
                f"{st.session_state.get('open_positions', 0)} 포지션"
            )
        with col4:
            st.metric(
                "승률",
                f"{st.session_state.get('win_rate', 0):.1f}%",
                f"총 {st.session_state.get('total_trades', 0)} 거래"
            )
        
        # 현재 포지션 요약
        st.subheader("📍 현재 포지션")
        if st.session_state.positions:
            position_df = pd.DataFrame(st.session_state.positions)
            position_df['수익률'] = position_df['unrealized_pnl_pct'].map('{:.2%}'.format)
            position_df['보유 시간'] = position_df['duration'].map('{:.1f}시간'.format)
            
            # 스타일이 적용된 데이터프레임
            st.dataframe(
                position_df[[
                    'symbol', 'side', 'entry_price', 'current_price',
                    'amount', 'unrealized_pnl', '수익률', '보유 시간'
                ]],
                use_container_width=True,
                height=200
            )
        else:
            st.info("현재 열린 포지션이 없습니다.")
        
        # 멀티 타임프레임 분석
        st.subheader("📊 멀티 타임프레임 분석")
        timeframes = ['5m', '15m', '1h', '4h']
        signals_df = pd.DataFrame({
            '시간프레임': timeframes,
            'RSI': np.random.randint(0, 100, len(timeframes)),
            'MACD': ['매수' if x > 50 else '매도' for x in np.random.randint(0, 100, len(timeframes))],
            'BB': ['상단', '중단', '하단', '중단'],
            '추세': ['상승', '상승', '하락', '하락'],
            '강도': np.random.randint(1, 10, len(timeframes))
        })
        st.dataframe(signals_df, use_container_width=True)
        
        # 시장 상황 요약
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 시장 동향")
            market_df = pd.DataFrame({
                '지표': ['변동성', '거래량', '추세 강도', '시장 상관성'],
                '상태': ['높음', '보통', '강함', '낮음'],
                '변화': ['↑', '→', '↑', '↓']
            })
            st.dataframe(market_df, use_container_width=True)
        
        with col2:
            st.subheader("⚡ 실시간 신호")
            signals_df = pd.DataFrame({
                '심볼': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
                '신호': ['매수', '관망', '매도'],
                '강도': ['강', '중', '약'],
                '시간': ['1분 전', '5분 전', '15분 전']
            })
            st.dataframe(signals_df, use_container_width=True)
    
    # 차트 탭
    with tabs[1]:
        st.header("📈 차트")
        
        # 차트 설정
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_symbol = st.selectbox(
                "심볼 선택",
                ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
                key="chart_symbol"
            )
        with col2:
            selected_timeframe = st.selectbox(
                "시간프레임",
                ["1m", "5m", "15m", "1h", "4h", "1d"],
                key="chart_timeframe"
            )
        with col3:
            selected_indicators = st.multiselect(
                "지표 선택",
                ["RSI", "MACD", "볼린저밴드", "이동평균선"],
                default=["RSI", "MACD"]
            )
        
        # 차트 표시
        if st.session_state.market_data is not None:
            fig = render_chart(st.session_state.market_data, selected_symbol, selected_indicators)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("차트 데이터를 불러오는 중입니다...")
    
    # 성과 탭
    with tabs[2]:
        st.header("💰 성과 분석")
        
        # 기간 선택
        period = st.selectbox(
            "기간 선택",
            ["전체", "오늘", "1주일", "1개월", "3개월", "6개월", "1년"]
        )
        
        # 성과 지표 표시
        if st.session_state.performance_report:
            render_performance_metrics(st.session_state.performance_report)
        else:
            st.info("성과 데이터가 없습니다.")
    
    # 포지션 탭
    with tabs[3]:
        st.header("📋 포지션 관리")
        
        # 현재 포지션
        st.subheader("📍 현재 포지션")
        if st.session_state.positions:
            for pos in st.session_state.positions:
                with st.expander(f"{pos['symbol']} {pos['side']} 포지션", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("진입가", f"${pos['entry_price']:,.2f}")
                        st.metric("현재가", f"${pos['current_price']:,.2f}")
                    with col2:
                        st.metric("수량", f"{pos['amount']:.4f}")
                        st.metric("레버리지", f"{pos.get('leverage', 1)}x")
                    with col3:
                        st.metric("미실현 손익", f"${pos['unrealized_pnl']:,.2f}")
                        st.metric("수익률", f"{pos['unrealized_pnl_pct']:.2%}")
                    
                    # 포지션 관리 버튼
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("전체 청산", key=f"close_{pos['symbol']}"):
                            asyncio.run(close_position(pos['symbol']))
                    with col2:
                        if st.button("부분 청산", key=f"partial_{pos['symbol']}"):
                            amount = st.number_input(
                                "청산할 수량",
                                min_value=0.0,
                                max_value=float(pos['amount']),
                                value=float(pos['amount'])/2,
                                step=0.001,
                                format="%.3f"
                            )
                            if st.button("확인", key=f"partial_confirm_{pos['symbol']}"):
                                asyncio.run(close_position(pos['symbol'], amount))
                    with col3:
                        if st.button("손절/익절 수정", key=f"sl_tp_{pos['symbol']}"):
                            current_price = float(pos['current_price'])
                            col1, col2 = st.columns(2)
                            with col1:
                                stop_loss = st.number_input(
                                    "손절가",
                                    value=float(pos.get('stop_loss', current_price * 0.95)),
                                    step=0.01,
                                    format="%.2f"
                                )
                            with col2:
                                take_profit = st.number_input(
                                    "익절가",
                                    value=float(pos.get('take_profit', current_price * 1.05)),
                                    step=0.01,
                                    format="%.2f"
                                )
                            if st.button("확인", key=f"sl_tp_confirm_{pos['symbol']}"):
                                asyncio.run(modify_position(
                                    pos['symbol'],
                                    stop_loss=stop_loss,
                                    take_profit=take_profit
                                ))
        else:
            st.info("현재 열린 포지션이 없습니다.")
        
        # 주문 내역
        st.subheader("📝 주문 내역")
        orders_df = pd.DataFrame({
            '시간': ['10:00:00', '10:05:00', '10:10:00'],
            '심볼': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            '유형': ['시장가', '지정가', '시장가'],
            '방향': ['매수', '매도', '매수'],
            '상태': ['체결', '대기', '체결'],
            '가격': ['$42,000', '$2,800', '$95']
        })
        st.dataframe(orders_df, use_container_width=True)
    
    # 거래 내역 탭
    with tabs[4]:
        st.header("📝 거래 내역")
        
        # 필터 설정
        col1, col2, col3 = st.columns(3)
        with col1:
            trade_symbol = st.selectbox(
                "심볼 선택",
                ["전체"] + list(set(t['symbol'] for t in st.session_state.trades))
                if st.session_state.trades else ["전체"]
            )
        with col2:
            trade_result = st.selectbox(
                "거래 결과",
                ["전체", "수익", "손실"]
            )
        with col3:
            trade_period = st.selectbox(
                "기간",
                ["전체", "오늘", "1주일", "1개월", "3개월"]
            )
        
        # 거래 내역 필터링 및 표시
        if st.session_state.trades:
            filtered_trades = filter_trades(
                st.session_state.trades,
                symbol=trade_symbol if trade_symbol != "전체" else None,
                result=trade_result if trade_result != "전체" else None,
                period=trade_period if trade_period != "전체" else None
            )
            
            if filtered_trades:
                trades_df = pd.DataFrame(filtered_trades)
                trades_df['수익률'] = trades_df['pnl_pct'].map('{:.2%}'.format)
                trades_df['거래시간'] = trades_df['duration'].map('{:.1f}시간'.format)
                
                st.dataframe(
                    trades_df[[
                        'timestamp', 'symbol', 'side', 'entry_price',
                        'exit_price', 'amount', 'pnl', '수익률', '거래시간'
                    ]],
                    use_container_width=True
                )
                
                # 거래 통계
                st.subheader("📊 거래 통계")
                stats = calculate_trade_stats(filtered_trades)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("💹 수익성 분석")
                    profit_stats = pd.DataFrame({
                        '지표': [
                            '총 거래',
                            '승률',
                            '평균 수익',
                            '최대 수익',
                            '최대 손실',
                            '손익비'
                        ],
                        '값': [
                            f"{stats['total_trades']}건",
                            f"{stats['win_rate']:.1%}",
                            f"${stats['avg_profit']:,.2f}",
                            f"${stats['max_profit']:,.2f}",
                            f"${stats['max_loss']:,.2f}",
                            f"{stats['profit_factor']:.2f}"
                        ]
                    })
                    st.dataframe(profit_stats, use_container_width=True)
                
                with col2:
                    st.subheader("⏱️ 시간대별 분석")
                    trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
                    time_stats = trades_df.groupby(pd.cut(
                        trades_df['hour'],
                        bins=[0, 8, 16, 24],
                        labels=['아시아', '유럽', '미국']
                    )).agg({
                        'symbol': 'count',
                        'pnl': lambda x: (x > 0).mean()
                    }).reset_index()
                    
                    time_stats.columns = ['시간대', '거래수', '승률']
                    time_stats['승률'] = time_stats['승률'].map('{:.1%}'.format)
                    st.dataframe(time_stats, use_container_width=True)
            else:
                st.info("필터링된 거래 내역이 없습니다.")
        else:
            st.info("거래 내역이 없습니다.")
    
    # 알림 탭
    with tabs[5]:
        st.header("🔔 알림 센터")
        
        # 알림 설정
        with st.expander("⚙️ 알림 설정", expanded=False):
            telegram_enabled = st.checkbox(
                "텔레그램 알림 활성화",
                value=st.session_state.telegram_enabled
            )
            if telegram_enabled != st.session_state.telegram_enabled:
                st.session_state.telegram_enabled = telegram_enabled
                setup_telegram()
            
            if telegram_enabled:
                notification_types = st.multiselect(
                    "알림 유형 선택",
                    ["진입 신호", "청산 신호", "손절", "익절", "시장 급변", "일일 리포트"],
                    default=list(st.session_state.notification_types)
                )
                st.session_state.notification_types = set(notification_types)
                
                notification_interval = st.slider(
                    "최소 알림 간격 (분)",
                    0, 60, st.session_state.notification_interval
                )
                if notification_interval != st.session_state.notification_interval:
                    st.session_state.notification_interval = notification_interval
                    setup_telegram()
        
        # 알림 테스트
        if st.button("테스트 알림 전송"):
            asyncio.run(telegram_notifier.send_message(
                "🔔 테스트 알림입니다.",
                "test"
            ))
            st.success("테스트 알림이 전송되었습니다.")
        
        # 알림 내역
        st.subheader("📋 알림 내역")
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        
        alerts_df = pd.DataFrame(st.session_state.alerts)
        if not alerts_df.empty:
            st.dataframe(alerts_df, use_container_width=True)
        else:
            st.info("알림 내역이 없습니다.")
    
    # 실시간 업데이트
    if st.session_state.bot and st.session_state.bot.is_running:
        if st.session_state.last_update is None or \
           (datetime.now() - st.session_state.last_update).seconds >= 5:
            try:
                # 비동기 함수를 동기적으로 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # 데이터 업데이트
                loop.run_until_complete(update_market_data())
                
                # 이벤트 루프 종료
                loop.close()
                
                # 화면 갱신
                st.rerun()
            except Exception as e:
                st.error(f"데이터 업데이트 중 오류 발생: {str(e)}")
                logger.error(f"데이터 업데이트 중 오류 발생: {str(e)}")
                
                # 텔레그램 알림 전송
                asyncio.run(telegram_notifier.send_error(str(e)))

if __name__ == "__main__":
    init_session_state()
    main() 