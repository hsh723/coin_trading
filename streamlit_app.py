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

# 페이지 설정은 반드시 다른 Streamlit 명령어보다 먼저 와야 함
st.set_page_config(
    page_title="코인 자동매매 시스템",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 모듈 임포트
import sys
import os
import time
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yaml
import threading
from pathlib import Path
from dotenv import load_dotenv

# 모듈 경로 문제 해결을 위한 임시 조치
class TradingBot:
    """임시 TradingBot 클래스"""
    def __init__(self, *args, **kwargs):
        self.running = False
        self.status = "초기화"
        
    def start(self):
        self.running = True
        self.status = "실행 중"
        return True
        
    def stop(self):
        self.running = False
        self.status = "중지됨"
        return True
    
    def get_status(self):
        return {
            "running": self.running,
            "status": self.status,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 모듈 임포트
try:
    from src.utils.database import DatabaseManager
    from src.utils.auth import AuthManager
    from src.utils.logger import logger
    from src.exchange.binance import BinanceExchange
    from src.strategies.integrated import IntegratedStrategy
    from src.risk.manager import RiskManager
    from src.utils.telegram import TelegramNotifier
except ImportError as e:
    st.error(f"모듈 임포트 오류: {str(e)}")
    # 임시 대체 클래스 정의
    class BinanceExchange:
        """임시 BinanceExchange 클래스"""
        def __init__(self, *args, **kwargs):
            self.api_key = kwargs.get('api_key', '')
            self.api_secret = kwargs.get('api_secret', '')
            self.testnet = kwargs.get('testnet', True)
            
        def fetch_positions(self):
            """포지션 정보 조회"""
            return []  # 임시 구현
            
        def fetch_ohlcv(self, symbol, timeframe, limit):
            """OHLCV 데이터 조회"""
            return []  # 임시 구현
            
        def fetch_my_trades(self, symbol, limit):
            """거래 내역 조회"""
            return []  # 임시 구현
            
        def create_order(self, **kwargs):
            """주문 생성"""
            return True  # 임시 구현
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
    # 임시 TelegramNotifier 클래스 재구현
    class TelegramNotifier:
        """임시 구현된 TelegramNotifier"""
        def __init__(self, **kwargs):
            # 어떤 인자든 받을 수 있도록 **kwargs 사용
            self.config = kwargs
            st.toast("텔레그램 알림 시스템 초기화됨", icon="📱")
            
        def send_message(self, message):
            """메시지 전송 시뮬레이션"""
            st.toast(f"텔레그램: {message[:30]}...", icon="📱")
            return True

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

# 전역 변수
db = DatabaseManager()
auth = AuthManager()
trading_bot = TradingBot()
trading_thread = None
stop_trading = False
telegram = TelegramNotifier()  # 인자 없이 초기화

def init_session_state():
    """세션 상태 초기화"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "대시보드"
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "market_data" not in st.session_state:
        st.session_state.market_data = None
    if "trades" not in st.session_state:
        st.session_state.trades = []
    if "trading_status" not in st.session_state:
        st.session_state.trading_status = False
    if "last_auth_time" not in st.session_state:
        st.session_state.last_auth_time = None
    if "positions" not in st.session_state:
        st.session_state.positions = []
    if "performance" not in st.session_state:
        st.session_state.performance = {
            "daily_return": 0,
            "weekly_return": 0,
            "monthly_return": 0,
            "total_trades": 0,
            "total_pnl": 0
        }

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
    now = datetime.now()
    dates = pd.date_range(end=now, periods=100, freq='1H')
    base_price = 50000  # BTC/USDT 기준 가격
    
    data = {
        'timestamp': dates,
        'open': [base_price * (1 + 0.001 * i) for i in range(100)],
        'high': [base_price * (1 + 0.002 * i) for i in range(100)],
        'low': [base_price * (1 - 0.001 * i) for i in range(100)],
        'close': [base_price * (1 + 0.0005 * i) for i in range(100)],
        'volume': [1000 * (1 + 0.01 * i) for i in range(100)]
    }
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def update_market_data(exchange):
    """시장 데이터 업데이트"""
    try:
        with st.spinner("시장 데이터를 불러오는 중..."):
            # API에서 데이터 가져오기 시도
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', 100)
            
            if ohlcv:
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                st.session_state.market_data = df
                add_log("시장 데이터 업데이트 완료")
                return True
            else:
                # API 데이터가 없으면 샘플 데이터 사용
                st.session_state.market_data = get_sample_market_data()
                add_log("샘플 시장 데이터 사용", "WARNING")
                return True
                
    except Exception as e:
        error_msg = f"시장 데이터 업데이트 오류: {str(e)}"
        add_log(error_msg, "ERROR")
        
        # 오류 발생 시 샘플 데이터 사용
        st.session_state.market_data = get_sample_market_data()
        add_log("샘플 시장 데이터 사용", "WARNING")
        return True

def update_positions(exchange):
    """포지션 정보 업데이트"""
    try:
        positions = exchange.fetch_positions()
        st.session_state.positions = positions
        add_log("포지션 정보 업데이트 완료")
    except Exception as e:
        error_msg = f"포지션 업데이트 실패: {str(e)}"
        add_log(error_msg, "ERROR")

def update_trades(exchange):
    """거래 내역 업데이트"""
    try:
        trades = exchange.fetch_my_trades('BTC/USDT', limit=10)
        st.session_state.trades = trades
        add_log("거래 내역 업데이트 완료")
    except Exception as e:
        error_msg = f"거래 내역 업데이트 실패: {str(e)}"
        add_log(error_msg, "ERROR")

def update_performance():
    """성과 지표 업데이트"""
    try:
        # 세션 상태 초기화 확인
        if "trades" not in st.session_state:
            st.session_state.trades = []
            
        if st.session_state.trades:
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = today - timedelta(days=7)
            month_ago = today - timedelta(days=30)
            
            # 임시 성과 계산
            st.session_state.performance = {
                "daily_return": 0.0,
                "weekly_return": 0.0,
                "monthly_return": 0.0,
                "total_trades": len(st.session_state.trades),
                "total_pnl": 0.0
            }
            
            add_log("성과 지표 업데이트 완료")
    except Exception as e:
        error_msg = f"성과 지표 업데이트 실패: {str(e)}"
        add_log(error_msg, "ERROR")

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
            if st.session_state.market_data is not None:
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

def run_async(coroutine):
    """비동기 함수를 동기적으로 실행"""
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(coroutine)
        loop.close()
        return result
    except Exception as e:
        print(f"비동기 실행 오류: {e}")
        return None

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
    if st.session_state.market_data is not None:
        try:
            fig = go.Figure(data=[go.Candlestick(
                x=st.session_state.market_data.index,
                open=st.session_state.market_data['open'],
                high=st.session_state.market_data['high'],
                low=st.session_state.market_data['low'],
                close=st.session_state.market_data['close']
            )])
            
            fig.update_layout(
                title='BTC/USDT 캔들스틱 차트',
                yaxis_title='가격',
                xaxis_title='시간',
                template='plotly_dark',
                height=500,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis_rangeslider_visible=False
            )
            
            return fig
        except Exception as e:
            st.error(f"차트 생성 오류: {str(e)}")
            return None
    else:
        st.warning("시장 데이터가 없습니다.")
        return None

def main_dashboard():
    """메인 대시보드"""
    # 상단 네비게이션 바
    col1, col2, col3 = st.columns([6, 1, 1])
    with col1:
        st.title("📈 암호화폐 트레이딩 봇")
    with col2:
        if st.button("🔄 새로고침"):
            st.rerun()
    with col3:
        if st.button("🚪 로그아웃"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
    
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
        st.metric("일일 수익률", f"{st.session_state.performance['daily_return']:.2f}%")
    
    if cols > 1:
        with metric_cols[1]:
            st.metric("주간 수익률", f"{st.session_state.performance['weekly_return']:.2f}%")
            st.metric("월간 수익률", f"{st.session_state.performance['monthly_return']:.2f}%")
        
        with metric_cols[2]:
            st.metric("총 수익", f"${st.session_state.performance['total_pnl']:.2f}")
            if st.session_state.positions:
                st.metric("현재 포지션", st.session_state.positions[0].get('symbol', 'N/A'))
    
    # 캔들스틱 차트
    st.header("차트")
    with st.spinner("차트를 불러오는 중..."):
        fig = render_chart()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("차트 데이터를 불러오는 중입니다...")
    
    # 거래 내역
    st.header("거래 내역")
    if st.session_state.trades:
        df = pd.DataFrame(st.session_state.trades)
        st.dataframe(df, use_container_width=True)
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

def login_form():
    """로그인 폼 표시"""
    st.title("🔒 로그인")
    
    with st.form("login_form"):
        username = st.text_input("사용자 이름")
        password = st.text_input("비밀번호", type="password")
        submit = st.form_submit_button("로그인")
        
        if submit:
            # 간단한 예시 - 실제로는 더 안전한 인증 로직이 필요함
            if username == "admin" and password == "password":
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("로그인 성공!")
                st.rerun()
            else:
                st.error("사용자 이름 또는 비밀번호가 올바르지 않습니다.")
    
    st.info("기본 계정: 사용자 이름 - admin, 비밀번호 - password")

def require_reauth():
    """재인증이 필요한지 확인"""
    # 재인증이 필요하지 않음을 나타내는 임시 구현
    return False

def reauth_form():
    """재인증 폼 표시"""
    st.warning("세션이 만료되었습니다. 다시 로그인해주세요.")
    
    with st.form("reauth_form"):
        password = st.text_input("비밀번호 확인", type="password")
        submit = st.form_submit_button("확인")
        
        if submit:
            # 간단한 예시 - 실제로는 더 안전한 인증 로직이 필요함
            if password == "password":
                st.session_state.authenticated = True
                st.success("인증되었습니다!")
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")

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