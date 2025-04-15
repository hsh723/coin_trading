import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import os
import json
import time
import threading
import queue

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """
    실시간 모니터링 대시보드
    
    주요 기능:
    - 실시간 거래 데이터 시각화
    - 시스템 상태 모니터링
    - 성과 지표 표시
    - 리스크 메트릭 표시
    - 알림 및 경고 표시
    """
    
    def __init__(self,
                 data_queue: queue.Queue,
                 update_interval: int = 1,
                 save_dir: str = "./dashboard_data"):
        """
        대시보드 초기화
        
        Args:
            data_queue: 실시간 데이터 큐
            update_interval: 업데이트 간격 (초)
            save_dir: 데이터 저장 디렉토리
        """
        self.data_queue = data_queue
        self.update_interval = update_interval
        self.save_dir = save_dir
        
        # 데이터 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 대시보드 상태 변수
        self.running = False
        self.last_update = datetime.now()
        self.data = {
            'prices': [],
            'positions': [],
            'portfolio_values': [],
            'trades': [],
            'metrics': {},
            'alerts': []
        }
        
        # Streamlit 설정
        st.set_page_config(
            page_title="암호화폐 트레이딩 모니터링",
            page_icon="📊",
            layout="wide"
        )
    
    def start(self):
        """대시보드 시작"""
        logger.info("모니터링 대시보드 시작 중...")
        self.running = True
        
        # 데이터 업데이트 스레드 시작
        self.update_thread = threading.Thread(target=self._update_data)
        self.update_thread.start()
        
        # 대시보드 렌더링 시작
        self._render_dashboard()
    
    def stop(self):
        """대시보드 중지"""
        logger.info("모니터링 대시보드 중지 중...")
        self.running = False
        
        # 스레드 종료 대기
        if hasattr(self, 'update_thread'):
            self.update_thread.join()
    
    def _update_data(self):
        """실시간 데이터 업데이트"""
        while self.running:
            try:
                # 데이터 큐에서 새로운 데이터 가져오기
                while not self.data_queue.empty():
                    new_data = self.data_queue.get()
                    self._process_new_data(new_data)
                
                # 데이터 저장
                self._save_data()
                
                # 업데이트 간격 대기
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"데이터 업데이트 중 오류 발생: {e}")
    
    def _process_new_data(self, new_data: Dict[str, Any]):
        """새로운 데이터 처리"""
        try:
            # 가격 데이터 업데이트
            if 'price' in new_data:
                self.data['prices'].append({
                    'timestamp': datetime.now(),
                    'price': new_data['price']
                })
            
            # 포지션 데이터 업데이트
            if 'position' in new_data:
                self.data['positions'].append({
                    'timestamp': datetime.now(),
                    'position': new_data['position']
                })
            
            # 포트폴리오 가치 업데이트
            if 'portfolio_value' in new_data:
                self.data['portfolio_values'].append({
                    'timestamp': datetime.now(),
                    'value': new_data['portfolio_value']
                })
            
            # 거래 데이터 업데이트
            if 'trade' in new_data:
                self.data['trades'].append({
                    'timestamp': datetime.now(),
                    **new_data['trade']
                })
            
            # 성과 지표 업데이트
            if 'metrics' in new_data:
                self.data['metrics'].update(new_data['metrics'])
            
            # 알림 업데이트
            if 'alert' in new_data:
                self.data['alerts'].append({
                    'timestamp': datetime.now(),
                    **new_data['alert']
                })
            
            # 데이터 정리 (최근 1000개만 유지)
            for key in ['prices', 'positions', 'portfolio_values', 'trades', 'alerts']:
                if len(self.data[key]) > 1000:
                    self.data[key] = self.data[key][-1000:]
            
        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {e}")
    
    def _save_data(self):
        """데이터 저장"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 데이터 저장
            for key in self.data:
                if key in ['prices', 'positions', 'portfolio_values', 'trades', 'alerts']:
                    df = pd.DataFrame(self.data[key])
                    df.to_csv(os.path.join(self.save_dir, f"{key}_{timestamp}.csv"), index=False)
                elif key == 'metrics':
                    with open(os.path.join(self.save_dir, f"metrics_{timestamp}.json"), 'w') as f:
                        json.dump(self.data[key], f, indent=4, default=str)
            
        except Exception as e:
            logger.error(f"데이터 저장 중 오류 발생: {e}")
    
    def _render_dashboard(self):
        """대시보드 렌더링"""
        # 페이지 제목
        st.title("암호화폐 트레이딩 모니터링 대시보드")
        
        # 실시간 데이터 섹션
        st.header("실시간 데이터")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("현재 가격")
            if self.data['prices']:
                current_price = self.data['prices'][-1]['price']
                st.metric("BTC/USDT", f"${current_price:,.2f}")
        
        with col2:
            st.subheader("현재 포지션")
            if self.data['positions']:
                current_position = self.data['positions'][-1]['position']
                st.metric("포지션 크기", f"{current_position:.4f}")
        
        with col3:
            st.subheader("포트폴리오 가치")
            if self.data['portfolio_values']:
                current_value = self.data['portfolio_values'][-1]['value']
                st.metric("포트폴리오 가치", f"${current_value:,.2f}")
        
        # 차트 섹션
        st.header("차트")
        tab1, tab2, tab3 = st.tabs(["가격", "포지션", "포트폴리오"])
        
        with tab1:
            if self.data['prices']:
                df = pd.DataFrame(self.data['prices'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['price'], name='가격'))
                fig.update_layout(title="BTC/USDT 가격", xaxis_title="시간", yaxis_title="가격")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if self.data['positions']:
                df = pd.DataFrame(self.data['positions'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['position'], name='포지션'))
                fig.update_layout(title="포지션 크기", xaxis_title="시간", yaxis_title="포지션")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if self.data['portfolio_values']:
                df = pd.DataFrame(self.data['portfolio_values'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['value'], name='포트폴리오 가치'))
                fig.update_layout(title="포트폴리오 가치", xaxis_title="시간", yaxis_title="가치")
                st.plotly_chart(fig, use_container_width=True)
        
        # 성과 지표 섹션
        st.header("성과 지표")
        if self.data['metrics']:
            metrics_df = pd.DataFrame([self.data['metrics']])
            st.dataframe(metrics_df)
        
        # 알림 섹션
        st.header("알림")
        if self.data['alerts']:
            alerts_df = pd.DataFrame(self.data['alerts'])
            st.dataframe(alerts_df)
        
        # 자동 새로고침
        time.sleep(self.update_interval)
        st.experimental_rerun() 