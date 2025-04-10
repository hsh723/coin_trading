import streamlit as st
import pandas as pd
from dataclasses import dataclass

@dataclass
class MarketStats:
    total_volume: float
    price_change: float
    market_sentiment: str
    top_movers: List[Dict]

class MarketOverviewDashboard:
    def __init__(self):
        self.refresh_interval = 60  # 1분
        
    async def render_market_overview(self, market_data: pd.DataFrame):
        """시장 개요 섹션 렌더링"""
        st.subheader("시장 개요")
        
        stats = await self._calculate_market_stats(market_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="24시간 거래량", 
                value=f"${stats.total_volume:,.0f}",
                delta=f"{stats.price_change:.2f}%"
            )
            
        with col2:
            st.metric(
                label="시장 상태",
                value=stats.market_sentiment
            )
            
        # Top Movers 테이블
        st.subheader("Top Movers")
        st.table(pd.DataFrame(stats.top_movers))
