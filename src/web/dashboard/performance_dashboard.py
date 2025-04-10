import streamlit as st
import pandas as pd
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    total_return: float
    daily_returns: pd.Series
    drawdown: float
    alpha: float
    beta: float
    sharpe: float

class PerformanceDashboard:
    def __init__(self):
        self.refresh_interval = 300  # 5분
        
    async def render_performance_dashboard(self, performance_data: Dict):
        """성과 대시보드 렌더링"""
        st.subheader("성과 분석")
        
        metrics = await self._calculate_performance_metrics(performance_data)
        
        # 주요 성과 지표
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="총 수익률",
                value=f"{metrics.total_return:.2f}%",
                delta=f"DD {metrics.drawdown:.1f}%"
            )
        with col2:
            st.metric(
                label="샤프 비율",
                value=f"{metrics.sharpe:.2f}"
            )
        with col3:
            st.metric(
                label="알파",
                value=f"{metrics.alpha:.3f}"
            )
            
        # 수익률 차트
        st.line_chart(metrics.daily_returns.cumsum())
