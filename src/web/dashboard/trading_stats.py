import streamlit as st
import pandas as pd
from dataclasses import dataclass

@dataclass
class TradingMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float
    daily_stats: Dict[str, float]
    performance_chart: pd.DataFrame

class TradingStatsDashboard:
    def __init__(self):
        self.refresh_interval = 300  # 5분
        
    async def render_trading_stats(self, trading_data: Dict):
        """거래 통계 섹션 렌더링"""
        st.subheader("거래 통계")
        
        metrics = await self._calculate_trading_metrics(trading_data)
        
        # 성과 지표
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="총 거래 횟수",
                value=metrics.total_trades,
                delta=f"승률 {metrics.win_rate:.1f}%"
            )
            
        with col2:
            st.metric(
                label="수익 팩터",
                value=f"{metrics.profit_factor:.2f}"
            )
            
        # 성과 차트
        st.line_chart(metrics.performance_chart)
