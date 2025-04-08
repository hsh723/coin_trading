"""
암호화폐 트레이딩 봇 웹 인터페이스
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import yaml
import json
from pathlib import Path
import asyncio
from typing import Dict, List, Any, Optional
import sys
from dotenv import load_dotenv
import nest_asyncio
import os
import numpy as np

from src.analysis.technical_analyzer import TechnicalAnalyzer
from src.analysis.self_learning import SelfLearningSystem
from src.strategy.portfolio_manager import PortfolioManager
from src.backtest.backtest_engine import BacktestEngine
from src.backtest.backtest_analyzer import BacktestAnalyzer
from src.dashboard.dashboard import Dashboard
from src.utils.config import load_config
from src.api.api_manager import APIManager
from src.backup.backup_manager import BackupManager
from src.optimization.optimizer import StrategyOptimizer, OptimizationResult
from src.notification.telegram_notifier import telegram_notifier
from src.notification.notification_manager import NotificationManager
from src.utils.performance_monitor import PerformanceMonitor, SystemMetrics
from src.strategy.base_strategy import BaseStrategy

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
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

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
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
        raise

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
        loop = get_or_create_eventloop()
        return loop.run_until_complete(async_func)
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

def render_backtest_tab():
    """백테스트 탭 렌더링"""
    st.header("🔄 백테스트")
    
    # 백테스트 설정
    with st.expander("⚙️ 백테스트 설정", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input(
                "시작일",
                value=datetime.now() - timedelta(days=180)
            )
        
        with col2:
            end_date = st.date_input(
                "종료일",
                value=datetime.now()
            )
        
        with col3:
            initial_capital = st.number_input(
                "초기 자본금",
                min_value=1000,
                value=10000,
                step=1000,
                format="%d"
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            commission = st.number_input(
                "수수료율",
                min_value=0.0,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f"
            )
        
        with col2:
            slippage = st.number_input(
                "슬리피지",
                min_value=0.0,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f"
            )
    
    # 전략 설정
    with st.expander("📊 전략 설정", expanded=True):
        strategy_params = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            strategy_params['rsi_period'] = st.number_input(
                "RSI 기간",
                min_value=1,
                max_value=100,
                value=14
            )
            
            strategy_params['rsi_upper'] = st.number_input(
                "RSI 상단",
                min_value=50,
                max_value=100,
                value=70
            )
            
            strategy_params['rsi_lower'] = st.number_input(
                "RSI 하단",
                min_value=0,
                max_value=50,
                value=30
            )
        
        with col2:
            strategy_params['ma_fast'] = st.number_input(
                "단기 이동평균",
                min_value=1,
                max_value=100,
                value=10
            )
            
            strategy_params['ma_slow'] = st.number_input(
                "장기 이동평균",
                min_value=1,
                max_value=200,
                value=30
            )
    
    # 백테스트 실행
    if st.button("백테스트 실행"):
        with st.spinner("백테스트 실행 중..."):
            try:
                # 전략 초기화
                strategy = IntegratedStrategy()
                strategy.update_parameters(strategy_params)
                
                # 백테스트 엔진 초기화
                engine = BacktestEngine(
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    commission=commission,
                    slippage=slippage,
                    database_manager=database_manager
                )
                
                # 백테스트 실행
                result = engine.run()
                
                if result:
                    # 결과 저장
                    st.session_state.backtest_result = result
                    
                    # 분석기 초기화
                    analyzer = BacktestAnalyzer(result)
                    
                    # 요약 통계
                    st.subheader("📊 백테스트 결과")
                    stats = analyzer.generate_summary_stats()
                    st.dataframe(stats, use_container_width=True)
                    
                    # 차트
                    charts = analyzer.plot_all()
                    
                    # 자본금 곡선
                    st.plotly_chart(
                        charts['equity_curve'],
                        use_container_width=True
                    )
                    
                    # 낙폭 차트
                    st.plotly_chart(
                        charts['drawdown'],
                        use_container_width=True
                    )
                    
                    # 월별 수익률
                    st.plotly_chart(
                        charts['monthly_returns'],
                        use_container_width=True
                    )
                    
                    # 거래 분석
                    st.plotly_chart(
                        charts['trade_analysis'],
                        use_container_width=True
                    )
                    
                    # 거래 내역
                    st.subheader("📝 거래 내역")
                    trades = analyzer.generate_trade_history()
                    if not trades.empty:
                        st.dataframe(trades, use_container_width=True)
                    else:
                        st.info("거래 내역이 없습니다.")
                    
                    # 결과 저장
                    if st.button("결과 저장"):
                        # 결과를 CSV로 저장
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        result_dir = "backtest_results"
                        os.makedirs(result_dir, exist_ok=True)
                        
                        # 요약 통계 저장
                        stats.to_csv(
                            f"{result_dir}/stats_{timestamp}.csv",
                            index=False,
                            encoding='utf-8-sig'
                        )
                        
                        # 거래 내역 저장
                        if not trades.empty:
                            trades.to_csv(
                                f"{result_dir}/trades_{timestamp}.csv",
                                index=False,
                                encoding='utf-8-sig'
                            )
                        
                        # 자본금 곡선 저장
                        result.equity_curve.to_csv(
                            f"{result_dir}/equity_{timestamp}.csv",
                            encoding='utf-8-sig'
                        )
                        
                        st.success("백테스트 결과가 저장되었습니다.")
                else:
                    st.error("백테스트 실행 실패")
            
            except Exception as e:
                st.error(f"백테스트 중 오류 발생: {str(e)}")
                logger.error(f"백테스트 중 오류 발생: {str(e)}")

def render_api_tab(api_manager: APIManager):
    """API 탭 렌더링"""
    st.header("API 통합")
    
    # 거래소 선택
    exchange = st.selectbox(
        "거래소 선택",
        ["binance", "bybit", "kucoin", "okx", "gateio"]
    )
    
    # 심볼 선택
    symbol = st.text_input("심볼", "BTC/USDT")
    
    # API 데이터 조회
    if st.button("데이터 조회"):
        try:
            # 시장 데이터
            market_data = asyncio.run(api_manager.get_market_data(symbol, exchange_id=exchange))
            if market_data:
                df = pd.DataFrame([vars(md) for md in market_data])
                st.subheader("시장 데이터")
                st.dataframe(df)
                
                # 캔들스틱 차트
                fig = go.Figure(data=[go.Candlestick(
                    x=df['timestamp'],
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close']
                )])
                st.plotly_chart(fig)
            
            # 호가 데이터
            orderbook = asyncio.run(api_manager.get_order_book(symbol, exchange_id=exchange))
            if orderbook:
                st.subheader("호가 데이터")
                st.write(f"스프레드: {orderbook.spread:.2f}")
                
                # 호가 차트
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[bid[0] for bid in orderbook.bids],
                    y=[bid[1] for bid in orderbook.bids],
                    name='매수',
                    marker_color='green'
                ))
                fig.add_trace(go.Bar(
                    x=[ask[0] for ask in orderbook.asks],
                    y=[ask[1] for ask in orderbook.asks],
                    name='매도',
                    marker_color='red'
                ))
                st.plotly_chart(fig)
            
            # 자금 조달 비율
            funding_rate = asyncio.run(api_manager.get_funding_rate(symbol, exchange_id=exchange))
            if funding_rate:
                st.subheader("자금 조달 비율")
                st.write(f"{funding_rate:.4%}")
            
            # 미체결약정
            open_interest = asyncio.run(api_manager.get_open_interest(symbol, exchange_id=exchange))
            if open_interest:
                st.subheader("미체결약정")
                st.write(f"{open_interest:,.2f}")
            
            # 청산 데이터
            liquidation = asyncio.run(api_manager.get_liquidation(symbol, exchange_id=exchange))
            if liquidation:
                st.subheader("청산 데이터")
                st.write(f"{liquidation:,.2f}")
            
            # 뉴스 데이터
            news = asyncio.run(api_manager.get_news(symbol))
            if news:
                st.subheader("뉴스")
                for article in news:
                    st.write(f"**{article['title']}**")
                    st.write(article['description'])
                    st.write(f"출처: {article['source']['name']}")
                    st.write("---")
            
            # 시장 감성 분석
            sentiment = asyncio.run(api_manager.get_market_sentiment(symbol))
            if sentiment:
                st.subheader("시장 감성 분석")
                fig = go.Figure(data=[
                    go.Bar(
                        x=['긍정', '부정', '중립'],
                        y=[sentiment['positive'], sentiment['negative'], sentiment['neutral']],
                        marker_color=['green', 'red', 'gray']
                    )
                ])
                st.plotly_chart(fig)
                
        except Exception as e:
            st.error(f"데이터 조회 중 오류 발생: {str(e)}")

def render_backup_tab(backup_manager: BackupManager):
    """백업 및 복구 탭 렌더링"""
    st.header("백업 및 복구")
    
    # 백업 생성
    st.subheader("백업 생성")
    col1, col2 = st.columns(2)
    
    with col1:
        include_database = st.checkbox("데이터베이스 포함", value=True)
        include_config = st.checkbox("설정 파일 포함", value=True)
    
    with col2:
        include_logs = st.checkbox("로그 파일 포함", value=True)
        include_strategies = st.checkbox("전략 파일 포함", value=True)
    
    if st.button("백업 생성"):
        try:
            backup_name = asyncio.run(
                backup_manager.create_backup(
                    include_database=include_database,
                    include_config=include_config,
                    include_logs=include_logs,
                    include_strategies=include_strategies
                )
            )
            st.success(f"백업 생성 완료: {backup_name}")
        except Exception as e:
            st.error(f"백업 생성 중 오류 발생: {str(e)}")
    
    # 백업 목록
    st.subheader("백업 목록")
    try:
        backups = asyncio.run(backup_manager.list_backups())
        
        if backups:
            # 백업 목록을 DataFrame으로 변환
            backup_data = []
            for backup in backups:
                backup_data.append({
                    '이름': backup['name'],
                    '생성 시간': backup['timestamp'],
                    '크기 (MB)': round(backup['size'] / (1024 * 1024), 2),
                    '데이터베이스': '✓' if backup['metadata']['include_database'] else '✗',
                    '설정 파일': '✓' if backup['metadata']['include_config'] else '✗',
                    '로그 파일': '✓' if backup['metadata']['include_logs'] else '✗',
                    '전략 파일': '✓' if backup['metadata']['include_strategies'] else '✗'
                })
            
            df = pd.DataFrame(backup_data)
            st.dataframe(df)
            
            # 백업 복구 및 삭제
            selected_backup = st.selectbox(
                "백업 선택",
                options=[backup['name'] for backup in backups],
                index=0
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("백업 복구"):
                    try:
                        asyncio.run(backup_manager.restore_backup(selected_backup))
                        st.success(f"백업 복구 완료: {selected_backup}")
                    except Exception as e:
                        st.error(f"백업 복구 중 오류 발생: {str(e)}")
            
            with col2:
                if st.button("백업 삭제"):
                    try:
                        asyncio.run(backup_manager.delete_backup(selected_backup))
                        st.success(f"백업 삭제 완료: {selected_backup}")
                    except Exception as e:
                        st.error(f"백업 삭제 중 오류 발생: {str(e)}")
        
        else:
            st.info("생성된 백업이 없습니다.")
            
    except Exception as e:
        st.error(f"백업 목록 조회 중 오류 발생: {str(e)}")

def render_optimization_tab(strategy: BaseStrategy):
    """최적화 탭 렌더링"""
    st.header("전략 최적화")
    
    # 최적화 설정
    st.subheader("최적화 설정")
    col1, col2 = st.columns(2)
    
    with col1:
        initial_capital = st.number_input(
            "초기 자본금",
            min_value=1000.0,
            value=10000.0,
            step=1000.0
        )
        commission = st.number_input(
            "수수료율",
            min_value=0.0,
            max_value=0.01,
            value=0.001,
            step=0.0001
        )
    
    with col2:
        n_iter = st.number_input(
            "반복 횟수",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        scoring_metric = st.selectbox(
            "점수 메트릭",
            options=['sharpe_ratio', 'total_return', 'profit_factor', 'win_rate', 'custom'],
            index=0
        )
    
    # 파라미터 그리드 설정
    st.subheader("파라미터 그리드")
    param_grid = {}
    
    for param in strategy.get_parameters():
        col1, col2 = st.columns(2)
        with col1:
            param_type = st.selectbox(
                f"{param} 타입",
                options=['list', 'range'],
                key=f"{param}_type"
            )
        with col2:
            if param_type == 'list':
                values = st.text_input(
                    f"{param} 값 (쉼표로 구분)",
                    key=f"{param}_list"
                )
                param_grid[param] = [float(x.strip()) for x in values.split(',')]
            else:
                min_val = st.number_input(
                    f"{param} 최소값",
                    key=f"{param}_min"
                )
                max_val = st.number_input(
                    f"{param} 최대값",
                    key=f"{param}_max"
                )
                param_grid[param] = (min_val, max_val)
    
    # 최적화 실행
    if st.button("최적화 실행"):
        try:
            # 최적화기 초기화
            optimizer = StrategyOptimizer(
                strategy=strategy,
                param_grid=param_grid,
                scoring_metric=scoring_metric,
                n_iter=n_iter
            )
            
            # 데이터 로드
            data = pd.read_csv("data/market_data.csv")
            
            # 최적화 실행
            with st.spinner("최적화 실행 중..."):
                result = asyncio.run(
                    optimizer.optimize(
                        data=data,
                        initial_capital=initial_capital,
                        commission=commission
                    )
                )
            
            # 결과 표시
            st.subheader("최적화 결과")
            
            # 최적 파라미터
            st.write("최적 파라미터:")
            st.json(result.best_params)
            
            # 성과 메트릭스
            st.write("성과 메트릭스:")
            metrics_df = pd.DataFrame([result.performance_metrics])
            st.dataframe(metrics_df)
            
            # 파라미터 중요도
            st.write("파라미터 중요도:")
            importance = optimizer._calculate_param_importance(result)
            importance_df = pd.DataFrame(
                list(importance.items()),
                columns=['파라미터', '중요도']
            )
            st.dataframe(importance_df)
            
            # 최적화 과정
            st.write("최적화 과정:")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result.optimization_history['iteration'],
                y=result.optimization_history['score'],
                mode='lines+markers',
                name='점수'
            ))
            fig.update_layout(
                title='최적화 과정',
                xaxis_title='반복',
                yaxis_title='점수'
            )
            st.plotly_chart(fig)
            
            # 결과 저장
            if st.button("결과 저장"):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    directory = f"optimization_results/{timestamp}"
                    optimizer.save_results(result, directory)
                    st.success(f"결과가 저장되었습니다: {directory}")
                except Exception as e:
                    st.error(f"결과 저장 중 오류 발생: {str(e)}")
            
        except Exception as e:
            st.error(f"최적화 중 오류 발생: {str(e)}")

def render_notification_tab(notification_manager: NotificationManager):
    """알림 탭 렌더링"""
    st.header("알림 설정")
    
    # 알림 규칙 관리
    st.subheader("알림 규칙 관리")
    
    # 새 규칙 추가
    with st.expander("새 규칙 추가", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            rule_name = st.text_input("규칙 이름")
            condition = st.text_area("조건 (Python 표현식)", help="예: data['price'] > 50000")
            message = st.text_area("메시지 템플릿", help="예: 가격이 {price}를 초과했습니다!")
        
        with col2:
            priority = st.number_input("우선순위", min_value=1, max_value=5, value=1)
            enabled = st.checkbox("활성화", value=True)
            notification_types = st.multiselect(
                "알림 유형",
                options=['telegram'],
                default=['telegram']
            )
        
        if st.button("규칙 추가"):
            try:
                if notification_manager.add_rule(
                    name=rule_name,
                    condition=condition,
                    message=message,
                    priority=priority,
                    enabled=enabled,
                    notification_types=notification_types
                ):
                    st.success("알림 규칙이 추가되었습니다.")
                else:
                    st.error("알림 규칙 추가에 실패했습니다.")
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
    
    # 규칙 목록
    st.subheader("규칙 목록")
    
    rules = list(notification_manager.rules.values())
    if rules:
        for rule in rules:
            with st.expander(f"{rule.name} ({'활성화' if rule.enabled else '비활성화'})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("조건:")
                    st.code(rule.condition)
                    st.write("메시지:")
                    st.code(rule.message)
                
                with col2:
                    st.write(f"우선순위: {rule.priority}")
                    st.write(f"알림 유형: {', '.join(rule.notification_types)}")
                    st.write(f"생성일: {rule.created_at}")
                    st.write(f"마지막 실행: {rule.last_triggered}")
                    st.write(f"실행 횟수: {rule.trigger_count}")
                
                if st.button("규칙 수정", key=f"edit_{rule.name}"):
                    st.session_state.editing_rule = rule.name
                
                if st.button("규칙 삭제", key=f"delete_{rule.name}"):
                    if notification_manager.remove_rule(rule.name):
                        st.success("규칙이 삭제되었습니다.")
                        st.experimental_rerun()
                    else:
                        st.error("규칙 삭제에 실패했습니다.")
    
    # 규칙 수정
    if hasattr(st.session_state, 'editing_rule'):
        rule = notification_manager.rules.get(st.session_state.editing_rule)
        if rule:
            with st.expander("규칙 수정", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    condition = st.text_area("조건", value=rule.condition)
                    message = st.text_area("메시지", value=rule.message)
                
                with col2:
                    priority = st.number_input("우선순위", value=rule.priority)
                    enabled = st.checkbox("활성화", value=rule.enabled)
                    notification_types = st.multiselect(
                        "알림 유형",
                        options=['telegram'],
                        default=rule.notification_types
                    )
                
                if st.button("수정 저장"):
                    if notification_manager.update_rule(
                        name=rule.name,
                        condition=condition,
                        message=message,
                        priority=priority,
                        enabled=enabled,
                        notification_types=notification_types
                    ):
                        st.success("규칙이 수정되었습니다.")
                        del st.session_state.editing_rule
                        st.experimental_rerun()
                    else:
                        st.error("규칙 수정에 실패했습니다.")
    
    # 알림 이력
    st.subheader("알림 이력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("시작일", value=datetime.now() - timedelta(days=7))
        rule_name = st.selectbox(
            "규칙 선택",
            options=['전체'] + [rule.name for rule in rules],
            index=0
        )
    
    with col2:
        end_date = st.date_input("종료일", value=datetime.now())
        if st.button("이력 조회"):
            history = notification_manager.get_notification_history(
                start_date=datetime.combine(start_date, datetime.min.time()),
                end_date=datetime.combine(end_date, datetime.max.time()),
                rule_name=rule_name if rule_name != '전체' else None
            )
            
            if history:
                df = pd.DataFrame(history)
                st.dataframe(df)
                
                if st.button("이력 삭제"):
                    if notification_manager.clear_notification_history(
                        start_date=datetime.combine(start_date, datetime.min.time()),
                        end_date=datetime.combine(end_date, datetime.max.time()),
                        rule_name=rule_name if rule_name != '전체' else None
                    ):
                        st.success("알림 이력이 삭제되었습니다.")
                        st.experimental_rerun()
                    else:
                        st.error("알림 이력 삭제에 실패했습니다.")
            else:
                st.info("조회된 알림 이력이 없습니다.")

def render_performance_tab(performance_monitor: PerformanceMonitor):
    """성능 모니터링 탭 렌더링"""
    st.header("성능 모니터링")
    
    # 현재 메트릭스
    st.subheader("현재 상태")
    current_metrics = performance_monitor.get_current_metrics()
    
    if current_metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "CPU 사용량",
                f"{current_metrics.cpu_usage:.1f}%",
                delta=None
            )
            st.metric(
                "메모리 사용량",
                f"{current_metrics.memory_usage:.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                "디스크 사용량",
                f"{current_metrics.disk_usage:.1f}%",
                delta=None
            )
            st.metric(
                "스왑 사용량",
                f"{current_metrics.swap_usage:.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "프로세스 수",
                f"{current_metrics.process_count}",
                delta=None
            )
            st.metric(
                "스레드 수",
                f"{current_metrics.thread_count}",
                delta=None
            )
        
        with col4:
            st.metric(
                "열린 파일 수",
                f"{current_metrics.open_files}",
                delta=None
            )
            st.metric(
                "네트워크 송신",
                f"{current_metrics.network_io['bytes_sent'] / (1024 * 1024):.1f} MB",
                delta=None
            )
    
    # 메트릭스 히스토리
    st.subheader("메트릭스 히스토리")
    metrics_history = performance_monitor.get_metrics_history()
    
    if metrics_history:
        df = pd.DataFrame([vars(m) for m in metrics_history])
        
        # CPU 및 메모리 사용량 차트
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['cpu_usage'],
                name='CPU 사용량',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['memory_usage'],
                name='메모리 사용량',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='CPU 및 메모리 사용량',
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 디스크 및 스왑 사용량 차트
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['disk_usage'],
                name='디스크 사용량',
                line=dict(color='green')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['swap_usage'],
                name='스왑 사용량',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='디스크 및 스왑 사용량',
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 네트워크 I/O 차트
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['network_io'].apply(lambda x: x['bytes_sent'] / (1024 * 1024)),
                name='송신 (MB)',
                line=dict(color='orange')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['network_io'].apply(lambda x: x['bytes_recv'] / (1024 * 1024)),
                name='수신 (MB)',
                line=dict(color='cyan')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='네트워크 I/O',
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 경고 메시지
    alerts = performance_monitor.check_alerts()
    if alerts:
        st.warning("시스템 경고:")
        for alert in alerts:
            st.write(f"- {alert}")
    
    # 메트릭스 관리
    st.subheader("메트릭스 관리")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("메트릭스 저장"):
            try:
                performance_monitor.save_metrics()
                st.success("메트릭스가 저장되었습니다.")
            except Exception as e:
                st.error(f"메트릭스 저장 중 오류 발생: {str(e)}")
    
    with col2:
        if st.button("메트릭스 초기화"):
            try:
                performance_monitor.clear_metrics()
                st.success("메트릭스가 초기화되었습니다.")
            except Exception as e:
                st.error(f"메트릭스 초기화 중 오류 발생: {str(e)}")

def main():
    """메인 함수"""
    try:
        # 이벤트 루프 초기화
        get_or_create_eventloop()
        
        # 설정 로드
        config = load_config()
        
        # API 관리자 초기화
        api_manager = APIManager(config)
        
        # 대시보드 초기화
        dashboard = Dashboard(config)
        
        # 백업 관리자 초기화
        backup_manager = BackupManager(database_manager=get_database_manager())
        
        # 성능 모니터링 초기화
        performance_monitor = PerformanceMonitor()
        performance_monitor.start()
        
        # 사이드바 설정
        st.sidebar.title("설정")
        
        # 탭 선택
        tab = st.sidebar.radio(
            "메뉴",
            ["대시보드", "백테스트", "API 통합", "백업 및 복구", "전략 최적화", "알림 설정", "성능 모니터링"]
        )
        
        # 선택된 탭 렌더링
        if tab == "대시보드":
            dashboard.render()
        elif tab == "백테스트":
            render_backtest_tab()
        elif tab == "API 통합":
            render_api_tab(api_manager)
        elif tab == "백업 및 복구":
            render_backup_tab(backup_manager)
        elif tab == "전략 최적화":
            render_optimization_tab(strategy)
        elif tab == "알림 설정":
            render_notification_tab(NotificationManager(
                database_manager=get_database_manager(),
                telegram_notifier=TelegramNotifier()
            ))
        elif tab == "성능 모니터링":
            render_performance_tab(performance_monitor)
            
    except Exception as e:
        logger.error(f"앱 실행 중 오류 발생: {str(e)}")
        st.error(f"오류 발생: {str(e)}")
    finally:
        # 성능 모니터링 중지
        if 'performance_monitor' in locals():
            performance_monitor.stop()

if __name__ == "__main__":
    init_session_state()
    main() 