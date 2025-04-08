import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ccxt
import os
from dotenv import load_dotenv
from strategies.integrated_strategy import IntegratedStrategy
from traders.integrated_trader import IntegratedTrader

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title="통합 트레이딩 시스템",
    page_icon="📈",
    layout="wide"
)

# 사이드바 설정
st.sidebar.title("설정")
symbol = st.sidebar.selectbox(
    "거래 심볼",
    ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
)
timeframe = st.sidebar.selectbox(
    "타임프레임",
    ["5m", "15m", "1h", "4h", "1d"]
)
risk_per_trade = st.sidebar.slider(
    "거래당 리스크 (%)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)

# 메인 컨텐츠
st.title("통합 트레이딩 시스템")

# 데이터 로드 및 차트 표시
@st.cache_data
def load_data(symbol, timeframe):
    exchange = ccxt.binance({
        'apiKey': os.getenv('BINANCE_API_KEY'),
        'secret': os.getenv('BINANCE_API_SECRET')
    })
    
    # 최근 100개의 캔들 데이터 가져오기
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# 전략 실행
def run_strategy(df):
    strategy = IntegratedStrategy()
    df = strategy.calculate_indicators(df)
    signals = strategy.generate_signal(df)
    return df, signals

# 차트 그리기
def plot_chart(df, signals):
    fig = go.Figure()
    
    # 캔들 차트
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ))
    
    # 이동평균선
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['ma20'],
        name='MA20',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['ma60'],
        name='MA60',
        line=dict(color='orange')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['ma200'],
        name='MA200',
        line=dict(color='red')
    ))
    
    # 볼린저 밴드
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['upper_band'],
        name='Upper Band',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['lower_band'],
        name='Lower Band',
        line=dict(color='gray', dash='dash')
    ))
    
    # 신호 포인트
    buy_signals = signals[signals['signal'] == 1]
    sell_signals = signals[signals['signal'] == -1]
    
    fig.add_trace(go.Scatter(
        x=buy_signals['timestamp'],
        y=buy_signals['close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=sell_signals['timestamp'],
        y=sell_signals['close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', size=10)
    ))
    
    fig.update_layout(
        title=f"{symbol} 차트",
        xaxis_title="시간",
        yaxis_title="가격",
        height=600
    )
    
    return fig

# 메인 실행
try:
    # 데이터 로드
    df = load_data(symbol, timeframe)
    
    # 전략 실행
    df, signals = run_strategy(df)
    
    # 차트 표시
    st.plotly_chart(plot_chart(df, signals), use_container_width=True)
    
    # 현재 포지션 정보
    st.subheader("현재 포지션")
    trader = IntegratedTrader(
        exchange=ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET')
        }),
        symbol=symbol,
        risk_per_trade=risk_per_trade
    )
    
    position = trader.get_current_position()
    if position:
        st.write(f"포지션: {position['side']}")
        st.write(f"진입가: {position['entry_price']}")
        st.write(f"현재가: {position['current_price']}")
        st.write(f"수익률: {position['pnl']}%")
    else:
        st.write("현재 포지션이 없습니다.")
    
    # 최근 거래 기록
    st.subheader("최근 거래 기록")
    recent_trades = trader.get_recent_trades(limit=5)
    if recent_trades:
        st.dataframe(pd.DataFrame(recent_trades))
    else:
        st.write("거래 기록이 없습니다.")
    
except Exception as e:
    st.error(f"오류 발생: {str(e)}") 