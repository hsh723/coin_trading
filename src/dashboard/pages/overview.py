"""
시스템 개요 페이지
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import psutil
import os

from src.utils.logger import setup_logger
from src.utils.performance import monitor_memory
from src.data.collector import DataCollector
from src.trading.executor import OrderExecutor

# 로거 설정
logger = setup_logger()

def get_system_info():
    """시스템 정보 조회"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory.percent,
        'disk_percent': disk.percent,
        'memory_used': memory.used / (1024**3),  # GB
        'disk_used': disk.used / (1024**3)  # GB
    }

def get_active_positions():
    """활성 포지션 조회"""
    try:
        executor = OrderExecutor(exchange=None, symbol='BTC/USDT', testnet=True)
        positions = executor.get_positions()
        return positions
    except Exception as e:
        logger.error(f"포지션 조회 오류: {str(e)}")
        return []

def main():
    st.title("시스템 개요")
    
    # 시스템 상태 카드
    col1, col2, col3 = st.columns(3)
    system_info = get_system_info()
    
    with col1:
        st.metric(
            "CPU 사용률",
            f"{system_info['cpu_percent']}%",
            f"{system_info['cpu_percent'] - 50}%" if system_info['cpu_percent'] > 50 else None
        )
    
    with col2:
        st.metric(
            "메모리 사용률",
            f"{system_info['memory_percent']}%",
            f"{system_info['memory_percent'] - 80}%" if system_info['memory_percent'] > 80 else None
        )
    
    with col3:
        st.metric(
            "디스크 사용률",
            f"{system_info['disk_percent']}%",
            f"{system_info['disk_percent'] - 80}%" if system_info['disk_percent'] > 80 else None
        )
    
    # 시스템 리소스 사용량 차트
    st.subheader("시스템 리소스 사용량")
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=system_info['cpu_percent'],
        title={'text': "CPU"},
        domain={'x': [0, 0.3], 'y': [0, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=system_info['memory_percent'],
        title={'text': "메모리"},
        domain={'x': [0.35, 0.65], 'y': [0, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=system_info['disk_percent'],
        title={'text': "디스크"},
        domain={'x': [0.7, 1], 'y': [0, 1]}
    ))
    
    fig.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # 활성 포지션
    st.subheader("활성 포지션")
    positions = get_active_positions()
    
    if positions:
        positions_df = pd.DataFrame(positions)
        st.dataframe(positions_df)
        
        # 포지션 분포 차트
        fig = go.Figure(data=[go.Pie(
            labels=positions_df['side'],
            values=positions_df['size'],
            hole=.3
        )])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("현재 활성 포지션이 없습니다.")
    
    # 시스템 로그
    st.subheader("시스템 로그")
    log_file = f'logs/trading_{datetime.now().strftime("%Y%m%d")}.log'
    
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.readlines()[-100:]  # 최근 100줄만 표시
            st.text_area("로그", ''.join(logs), height=300)
    else:
        st.info("로그 파일이 없습니다.")
    
    # 자동 새로고침
    if st.session_state.system_status == 'running':
        st.experimental_rerun()

if __name__ == "__main__":
    main() 