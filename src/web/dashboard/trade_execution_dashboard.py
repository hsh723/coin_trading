import streamlit as st
import pandas as pd
from dataclasses import dataclass

@dataclass
class ExecutionMetrics:
    active_orders: List[Dict]
    filled_orders: List[Dict]
    execution_stats: Dict[str, float]
    slippage_analysis: Dict[str, float]

class TradeExecutionDashboard:
    def __init__(self):
        self.refresh_interval = 10  # 10초
        
    async def render_execution_dashboard(self, execution_data: Dict):
        """거래 실행 대시보드 렌더링"""
        st.subheader("거래 실행 현황")
        
        metrics = await self._calculate_execution_metrics(execution_data)
        
        # 활성 주문
        st.write("활성 주문")
        active_orders_df = pd.DataFrame(metrics.active_orders)
        st.dataframe(active_orders_df)
        
        # 실행 통계
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="평균 슬리피지",
                value=f"{metrics.execution_stats['avg_slippage']:.4f}%"
            )
        with col2:
            st.metric(
                label="체결률",
                value=f"{metrics.execution_stats['fill_rate']:.1f}%"
            )
