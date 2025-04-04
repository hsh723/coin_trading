"""
단순화된 Streamlit 앱
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="코인 트레이딩 봇",
    page_icon="💰",
    layout="wide"
)

# 사이드바 설정
st.sidebar.title("설정")

# API 키 설정
api_key = st.sidebar.text_input("Binance API Key", type="password")
api_secret = st.sidebar.text_input("Binance API Secret", type="password")

# 거래 설정
symbol = st.sidebar.selectbox(
    "거래 심볼",
    ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
)

timeframe = st.sidebar.selectbox(
    "시간 프레임",
    ["1m", "5m", "15m", "1h", "4h", "1d"]
)

# 메인 컨텐츠
st.title("코인 트레이딩 봇")

# 상태 표시
status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.metric("현재 가격", "0.00 USDT")

with status_col2:
    st.metric("24시간 변동률", "0.00%")

with status_col3:
    st.metric("현재 포지션", "없음")

# 샘플 데이터 생성
def create_sample_data():
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
    
    return df

# 차트 표시
def render_chart(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # 캔들스틱 차트
    fig.add_trace(go.Candlestick(x=data['timestamp'],
                               open=data['open'],
                               high=data['high'],
                               low=data['low'],
                               close=data['close'],
                               name='OHLC'),
                 row=1, col=1)
    
    # 거래량 차트
    fig.add_trace(go.Bar(x=data['timestamp'], y=data['volume'], name='Volume'),
                 row=2, col=1)
    
    # 레이아웃 설정
    fig.update_layout(height=600, title_text=f"{symbol} {timeframe} 차트")
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

# 샘플 데이터 생성 및 차트 표시
data = create_sample_data()
chart = render_chart(data)
st.plotly_chart(chart, use_container_width=True)

# 거래 기록
st.subheader("거래 기록")
trades_df = pd.DataFrame({
    '시간': [datetime.now() - timedelta(hours=i) for i in range(5)],
    '종류': ['매수', '매도', '매수', '매도', '매수'],
    '가격': [50000, 51000, 50500, 51500, 52000],
    '수량': [0.1, 0.1, 0.2, 0.2, 0.3],
    '수익': [0, 100, 0, 200, 0]
})
st.dataframe(trades_df)

# 성능 분석
st.subheader("성능 분석")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("총 수익률", "15.5%")
    
with col2:
    st.metric("승률", "65%")
    
with col3:
    st.metric("평균 수익", "2.5%")
    
with col4:
    st.metric("샤프 비율", "1.8")

# 시스템 모니터링
st.subheader("시스템 모니터링")
monitoring_col1, monitoring_col2, monitoring_col3 = st.columns(3)

with monitoring_col1:
    st.metric("CPU 사용률", "45%")
    
with monitoring_col2:
    st.metric("메모리 사용률", "60%")
    
with monitoring_col3:
    st.metric("디스크 사용률", "30%")

if __name__ == "__main__":
    pass 