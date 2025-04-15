import streamlit as st
import logging
from datetime import datetime
import time
import threading
import queue

logger = logging.getLogger(__name__)

class SimpleDashboard:
    """
    간단한 대시보드
    
    주요 기능:
    - 당일 수익률 표시
    - 현재 포지션 상태 표시
    """
    
    def __init__(self, metrics_queue: queue.Queue):
        """
        대시보드 초기화
        
        Args:
            metrics_queue: 성과 지표 큐
        """
        self.metrics_queue = metrics_queue
        self.running = False
        
        # Streamlit 설정
        st.set_page_config(
            page_title="트레이딩 대시보드",
            page_icon="💰",
            layout="centered"
        )
    
    def start(self):
        """대시보드 시작"""
        logger.info("대시보드 시작 중...")
        self.running = True
        
        # 대시보드 렌더링 시작
        self._render_dashboard()
    
    def stop(self):
        """대시보드 중지"""
        logger.info("대시보드 중지 중...")
        self.running = False
    
    def _render_dashboard(self):
        """대시보드 렌더링"""
        # 페이지 제목
        st.title("트레이딩 대시보드")
        
        # 실시간 데이터 섹션
        st.header("실시간 성과")
        
        # 성과 지표 표시
        while self.running:
            try:
                # 성과 지표 가져오기
                if not self.metrics_queue.empty():
                    metrics = self.metrics_queue.get()
                    
                    # 당일 수익률 표시
                    daily_return = metrics.get('daily_return', 0.0)
                    st.metric(
                        "당일 수익률",
                        f"{daily_return*100:.2f}%",
                        delta=f"{daily_return*100:.2f}%"
                    )
                    
                    # 현재 포지션 상태 표시
                    position = metrics.get('current_position', 0.0)
                    st.metric(
                        "현재 포지션",
                        f"{position:.4f}",
                        delta="롱" if position > 0 else "숏" if position < 0 else "없음"
                    )
                
                # 업데이트 간격 대기
                time.sleep(1)
                st.experimental_rerun()
                
            except Exception as e:
                logger.error(f"대시보드 렌더링 중 오류 발생: {e}")
                time.sleep(1) 