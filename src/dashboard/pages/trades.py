"""
거래 내역 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

from src.utils.logger import setup_logger
from src.trading.executor import OrderExecutor

# 로거 설정
logger = setup_logger()

def get_trade_history():
    """거래 내역 조회"""
    try:
        executor = OrderExecutor(exchange=None, symbol='BTC/USDT', testnet=True)
        trades = executor.get_trade_history()
        return pd.DataFrame(trades)
    except Exception as e:
        logger.error(f"거래 내역 조회 오류: {str(e)}")
        return pd.DataFrame()

def analyze_trades(trades_df: pd.DataFrame):
    """거래 분석"""
    if len(trades_df) == 0:
        return {}
    
    # 시간대별 수익률
    trades_df['hour'] = pd.to_datetime(trades_df['timestamp']).dt.hour
    hourly_returns = trades_df.groupby('hour')['pnl'].mean()
    
    # 요일별 수익률
    trades_df['weekday'] = pd.to_datetime(trades_df['timestamp']).dt.day_name()
    weekday_returns = trades_df.groupby('weekday')['pnl'].mean()
    
    # 거래 크기별 수익률
    trades_df['size_category'] = pd.qcut(trades_df['size'], q=4, labels=['Small', 'Medium', 'Large', 'Very Large'])
    size_returns = trades_df.groupby('size_category')['pnl'].mean()
    
    return {
        'hourly_returns': hourly_returns,
        'weekday_returns': weekday_returns,
        'size_returns': size_returns
    }

def main():
    st.title("거래 내역")
    
    # 기간 선택
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "시작일",
            datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "종료일",
            datetime.now()
        )
    
    # 거래 내역 조회
    trades_df = get_trade_history()
    
    if not trades_df.empty:
        # 날짜 필터링
        trades_df['date'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df[
            (trades_df['date'].dt.date >= start_date) &
            (trades_df['date'].dt.date <= end_date)
        ]
        
        # 거래 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "총 거래 수",
                f"{len(trades_df):,}",
                f"{len(trades_df):,}"
            )
        with col2:
            st.metric(
                "평균 수익",
                f"${trades_df['pnl'].mean():.2f}",
                f"${trades_df['pnl'].mean():.2f}"
            )
        with col3:
            st.metric(
                "총 수익",
                f"${trades_df['pnl'].sum():.2f}",
                f"${trades_df['pnl'].sum():.2f}"
            )
        
        # 거래 분석
        analysis = analyze_trades(trades_df)
        
        # 시간대별 수익률
        st.subheader("시간대별 수익률")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=analysis['hourly_returns'].index,
            y=analysis['hourly_returns'].values,
            name='시간대별 수익률'
        ))
        fig.update_layout(
            title='시간대별 평균 수익률',
            xaxis_title='시간',
            yaxis_title='평균 수익률'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 요일별 수익률
        st.subheader("요일별 수익률")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=analysis['weekday_returns'].index,
            y=analysis['weekday_returns'].values,
            name='요일별 수익률'
        ))
        fig.update_layout(
            title='요일별 평균 수익률',
            xaxis_title='요일',
            yaxis_title='평균 수익률'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 거래 크기별 수익률
        st.subheader("거래 크기별 수익률")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=analysis['size_returns'].index,
            y=analysis['size_returns'].values,
            name='거래 크기별 수익률'
        ))
        fig.update_layout(
            title='거래 크기별 평균 수익률',
            xaxis_title='거래 크기',
            yaxis_title='평균 수익률'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 거래 내역 테이블
        st.subheader("상세 거래 내역")
        st.dataframe(trades_df)
    else:
        st.info("선택한 기간 동안의 거래 내역이 없습니다.")
    
    # 자동 새로고침
    if st.session_state.system_status == 'running':
        st.experimental_rerun()

if __name__ == "__main__":
    main() 