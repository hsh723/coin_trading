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

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
try:
    from src.utils.database import DatabaseManager
    from src.utils.auth import AuthManager
    from src.utils.logger import logger
    from src.trading_bot import TradingBot
    from src.exchange.binance import BinanceExchange
    from src.strategies.integrated import IntegratedStrategy
    from src.risk.manager import RiskManager
    from src.utils.telegram import TelegramNotifier
except ImportError as e:
    st.error(f"모듈 임포트 오류: {str(e)}")
    # 임시 대체 클래스 정의
    class TradingBot:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass
    class BinanceExchange:
        def __init__(self, *args, **kwargs):
            pass
        def fetch_ohlcv(self, *args, **kwargs):
            return []
        def fetch_positions(self):
            return []
        def fetch_my_trades(self, *args, **kwargs):
            return []
        def create_order(self, **kwargs):
            pass
    class IntegratedStrategy:
        def __init__(self):
            pass
        def generate_signal(self, *args):
            return None
        def calculate_position_size(self, *args):
            return 0
    class RiskManager:
        def __init__(self, *args, **kwargs):
            self.risk_per_trade = 0.02
        def get_capital(self):
            return 1000.0
    class TelegramNotifier:
        def __init__(self, *args, **kwargs):
            pass
        def send_message(self, *args):
            pass

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="암호화폐 트레이딩 봇",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# 전역 변수
db = DatabaseManager()
auth = AuthManager()
trading_bot = None
trading_thread = None
stop_trading = False
telegram = TelegramNotifier(
    token=os.getenv("TELEGRAM_BOT_TOKEN"),
    chat_id=os.getenv("TELEGRAM_CHAT_ID")
)

def init_session_state():
    """세션 상태 초기화"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'trading_status' not in st.session_state:
        st.session_state.trading_status = False
    if 'last_auth_time' not in st.session_state:
        st.session_state.last_auth_time = None
    if 'market_data' not in st.session_state:
        st.session_state.market_data = pd.DataFrame()
    if 'positions' not in st.session_state:
        st.session_state.positions = pd.DataFrame()
    if 'trades' not in st.session_state:
        st.session_state.trades = pd.DataFrame()
    if 'performance' not in st.session_state:
        st.session_state.performance = {
            'daily_return': 0,
            'weekly_return': 0,
            'monthly_return': 0,
            'total_trades': 0,
            'total_pnl': 0
        }
    if 'logs' not in st.session_state:
        st.session_state.logs = []

def add_log(message: str, level: str = "INFO"):
    """로그 추가"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {level}: {message}"
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

def update_market_data(exchange):
    """시장 데이터 업데이트"""
    try:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', 100)
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        st.session_state.market_data = df
        add_log("시장 데이터 업데이트 완료")
    except Exception as e:
        error_msg = f"시장 데이터 업데이트 실패: {str(e)}"
        add_log(error_msg, "ERROR")

def update_positions(exchange):
    """포지션 정보 업데이트"""
    try:
        positions = exchange.fetch_positions()
        if positions:
            df = pd.DataFrame(positions)
            st.session_state.positions = df
            add_log("포지션 정보 업데이트 완료")
    except Exception as e:
        error_msg = f"포지션 업데이트 실패: {str(e)}"
        add_log(error_msg, "ERROR")

def update_trades(exchange):
    """거래 내역 업데이트"""
    try:
        trades = exchange.fetch_my_trades('BTC/USDT', limit=10)
        if trades:
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            st.session_state.trades = df
            add_log("거래 내역 업데이트 완료")
    except Exception as e:
        error_msg = f"거래 내역 업데이트 실패: {str(e)}"
        add_log(error_msg, "ERROR")

def update_performance():
    """성과 지표 업데이트"""
    try:
        if not st.session_state.trades.empty:
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = today - timedelta(days=7)
            month_ago = today - timedelta(days=30)
            
            recent_trades = st.session_state.trades
            
            # 수익률 계산
            daily_pnl = recent_trades[recent_trades['timestamp'] >= today]['pnl'].sum()
            weekly_pnl = recent_trades[recent_trades['timestamp'] >= week_ago]['pnl'].sum()
            monthly_pnl = recent_trades[recent_trades['timestamp'] >= month_ago]['pnl'].sum()
            
            st.session_state.performance = {
                'daily_return': f"{daily_pnl:.2f}%",
                'weekly_return': f"{weekly_pnl:.2f}%",
                'monthly_return': f"{monthly_pnl:.2f}%",
                'total_trades': len(recent_trades),
                'total_pnl': f"${recent_trades['pnl'].sum():.2f}"
            }
    except Exception as e:
        logger.error(f"성과 지표 업데이트 실패: {str(e)}")

def trading_loop(exchange, strategy, risk_manager):
    """트레이딩 루프"""
    global stop_trading
    
    while not stop_trading:
        try:
            # 데이터 업데이트
            update_market_data(exchange)
            update_positions(exchange)
            update_trades(exchange)
            update_performance()
            
            # 신호 생성
            if not st.session_state.market_data.empty:
                signal = strategy.generate_signal(st.session_state.market_data)
                
                if signal:
                    # 포지션 크기 계산
                    position_size = strategy.calculate_position_size(
                        risk_manager.get_capital(),
                        risk_manager.risk_per_trade,
                        signal['price'],
                        signal['stop_loss']
                    )
                    
                    # 주문 실행
                    order = {
                        'symbol': signal['symbol'],
                        'type': signal['type'],
                        'side': signal['side'],
                        'amount': position_size,
                        'price': signal['price']
                    }
                    
                    exchange.create_order(**order)
                    log_msg = f"주문 실행: {order}"
                    add_log(log_msg)
                    telegram.send_message(log_msg)
            
            time.sleep(60)  # 1분 대기
            
        except Exception as e:
            error_msg = f"트레이딩 에러: {str(e)}"
            add_log(error_msg, "ERROR")
            time.sleep(60)

def start_trading():
    """트레이딩 시작"""
    global trading_thread, stop_trading
    
    if trading_thread and trading_thread.is_alive():
        st.error("트레이딩이 이미 실행 중입니다.")
        return
    
    config = load_trading_config()
    
    exchange = BinanceExchange(
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        testnet=True
    )
    
    strategy = IntegratedStrategy()
    
    risk_manager = RiskManager(
        initial_capital=float(config.get('max_position_size', 100.0)),
        risk_per_trade=float(config.get('stop_loss', 2.0)) / 100,
        max_positions=3,
        daily_loss_limit=0.05,
        max_drawdown=0.10
    )
    
    stop_trading = False
    trading_thread = threading.Thread(
        target=trading_loop,
        args=(exchange, strategy, risk_manager)
    )
    trading_thread.start()
    
    st.session_state.trading_status = True
    success_msg = "트레이딩이 시작되었습니다."
    st.success(success_msg)
    add_log(success_msg)
    telegram.send_message(success_msg)

def stop_trading_loop():
    """트레이딩 중지"""
    global trading_thread, stop_trading
    
    if trading_thread and trading_thread.is_alive():
        stop_trading = True
        trading_thread.join()
        st.session_state.trading_status = False
        success_msg = "트레이딩이 중지되었습니다."
        st.success(success_msg)
        add_log(success_msg)
        telegram.send_message(success_msg)
    else:
        warning_msg = "실행 중인 트레이딩이 없습니다."
        st.warning(warning_msg)
        add_log(warning_msg, "WARNING")

def render_chart():
    """차트 렌더링"""
    if not st.session_state.market_data.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=st.session_state.market_data.index,
            open=st.session_state.market_data['open'],
            high=st.session_state.market_data['high'],
            low=st.session_state.market_data['low'],
            close=st.session_state.market_data['close']
        )])
        
        fig.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False
        )
        
        return fig
    return None

def main_dashboard():
    """메인 대시보드"""
    # 상단 네비게이션 바
    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        st.title("📈 암호화폐 트레이딩 봇")
    with col2:
        if st.button("🔄 새로고침"):
            st.experimental_rerun()
    with col3:
        if st.button("🚪 로그아웃"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.experimental_rerun()
    
    # 거래 상태 및 제어
    st.header("거래 상태")
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        st.metric("현재 상태", "실행 중" if st.session_state.trading_status else "중지됨")
    
    with status_col2:
        if st.session_state.trading_status:
            if st.button("⏹️ 거래 중지", key="stop_trading"):
                stop_trading_loop()
        else:
            if st.button("▶️ 거래 시작", key="start_trading"):
                start_trading()
    
    # 실시간 거래 정보
    st.header("실시간 정보")
    
    # 모바일에서도 보기 좋게 컬럼 조정
    if st.checkbox("모바일 뷰", value=False):
        cols = 1
    else:
        cols = 3
    
    metric_cols = st.columns(cols)
    
    with metric_cols[0]:
        st.metric("총 거래 횟수", st.session_state.performance['total_trades'])
        st.metric("일일 수익률", st.session_state.performance['daily_return'])
    
    if cols > 1:
        with metric_cols[1]:
            st.metric("주간 수익률", st.session_state.performance['weekly_return'])
            st.metric("월간 수익률", st.session_state.performance['monthly_return'])
        
        with metric_cols[2]:
            st.metric("총 수익", st.session_state.performance['total_pnl'])
            if not st.session_state.positions.empty:
                st.metric("현재 포지션", st.session_state.positions.iloc[0]['symbol'])
    
    # 캔들스틱 차트
    st.header("차트")
    fig = render_chart()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("차트 데이터를 불러오는 중입니다...")
    
    # 거래 내역
    st.header("거래 내역")
    if not st.session_state.trades.empty:
        st.dataframe(st.session_state.trades, use_container_width=True)
    else:
        st.info("거래 내역이 없습니다.")
    
    # 로그 표시
    st.header("시스템 로그")
    st.text_area("로그", "\n".join(st.session_state.logs[-50:]), height=200)

def settings_page():
    """설정 페이지"""
    st.header("⚙️ 설정")
    
    # 설정 로드
    config = load_trading_config()
    
    # RSI 설정
    st.subheader("RSI 설정")
    rsi_period = st.number_input(
        "RSI 기간",
        min_value=5,
        max_value=50,
        value=config.get("rsi_period", 14),
        step=1
    )
    rsi_overbought = st.number_input(
        "과매수 기준",
        min_value=50,
        max_value=100,
        value=config.get("rsi_overbought", 70),
        step=1
    )
    rsi_oversold = st.number_input(
        "과매도 기준",
        min_value=0,
        max_value=50,
        value=config.get("rsi_oversold", 30),
        step=1
    )
    
    # 볼린저 밴드 설정
    st.subheader("볼린저 밴드 설정")
    bb_period = st.number_input(
        "볼린저 밴드 기간",
        min_value=5,
        max_value=50,
        value=config.get("bb_period", 20),
        step=1
    )
    bb_std = st.number_input(
        "표준편차",
        min_value=1.0,
        max_value=3.0,
        value=config.get("bb_std", 2.0),
        step=0.1
    )
    
    # 리스크 관리 설정
    st.subheader("리스크 관리")
    max_position_size = st.number_input(
        "최대 포지션 크기 (USDT)",
        min_value=10.0,
        max_value=10000.0,
        value=config.get("max_position_size", 100.0),
        step=10.0
    )
    stop_loss = st.number_input(
        "손절 비율 (%)",
        min_value=0.1,
        max_value=10.0,
        value=config.get("stop_loss", 2.0),
        step=0.1
    )
    take_profit = st.number_input(
        "익절 비율 (%)",
        min_value=0.1,
        max_value=20.0,
        value=config.get("take_profit", 5.0),
        step=0.1
    )
    
    # 설정 저장
    if st.button("💾 설정 저장"):
        new_config = {
            "rsi_period": rsi_period,
            "rsi_overbought": rsi_overbought,
            "rsi_oversold": rsi_oversold,
            "bb_period": bb_period,
            "bb_std": bb_std,
            "max_position_size": max_position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }
        
        save_trading_config(new_config)
        st.success("설정이 저장되었습니다.")

def main():
    """메인 함수"""
    init_session_state()
    
    if not st.session_state.authenticated:
        login_form()
    else:
        if require_reauth():
            reauth_form()
        else:
            # 사이드바 메뉴
            st.sidebar.title("메뉴")
            menu = st.sidebar.radio(
                "선택",
                ["대시보드", "설정"],
                format_func=lambda x: "📊 " + x if x == "대시보드" else "⚙️ " + x
            )
            
            if menu == "대시보드":
                main_dashboard()
            elif menu == "설정":
                settings_page()

if __name__ == "__main__":
    main() 