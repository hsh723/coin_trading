import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass

@dataclass
class OrderBookVisualization:
    bids: pd.DataFrame
    asks: pd.DataFrame
    spread: float
    depth_stats: Dict[str, float]

class OrderBookDashboard:
    def __init__(self):
        self.refresh_interval = 1  # 1초
        
    async def render_order_book(self, order_book_data: Dict):
        """주문장 대시보드 렌더링"""
        st.subheader("실시간 주문장")
        
        vis_data = await self._prepare_visualization(order_book_data)
        
        # 스프레드 정보
        st.metric(
            label="현재 스프레드",
            value=f"{vis_data.spread:.4f}%"
        )
        
        # 주문장 차트
        fig = self._create_order_book_chart(vis_data)
        st.plotly_chart(fig, use_container_width=True)
