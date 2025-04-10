import streamlit as st
import pandas as pd
from dataclasses import dataclass

@dataclass
class PortfolioMetrics:
    total_value: float
    pnl: float
    positions: List[Dict]
    risk_metrics: Dict[str, float]

class PortfolioDashboard:
    def __init__(self):
        self.refresh_interval = 30  # 30초
        
    async def render_portfolio(self, portfolio_data: Dict):
        """포트폴리오 섹션 렌더링"""
        st.subheader("포트폴리오 현황")
        
        metrics = await self._calculate_portfolio_metrics(portfolio_data)
        
        # 성과 지표
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="포트폴리오 가치",
                value=f"${metrics.total_value:,.2f}",
                delta=f"{metrics.pnl:.2f}%"
            )
            
        # 포지션 테이블
        st.subheader("활성 포지션")
        positions_df = pd.DataFrame(metrics.positions)
        st.dataframe(positions_df)
