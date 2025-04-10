import streamlit as st
import pandas as pd
from dataclasses import dataclass

@dataclass
class TradeHistoryMetrics:
    total_trades: int
    win_rate: float
    profit_loss: Dict[str, float]
    trade_details: pd.DataFrame

class TradeHistoryDashboard:
    def __init__(self):
        self.page_size = 20
        
    async def render_trade_history(self, trade_data: Dict):
        """거래 기록 대시보드 렌더링"""
        st.subheader("거래 기록")
        
        metrics = await self._calculate_trade_metrics(trade_data)
        
        # 거래 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="총 거래 수",
                value=metrics.total_trades,
                delta=f"승률 {metrics.win_rate:.1f}%"
            )
            
        # 거래 내역 테이블
        st.dataframe(
            metrics.trade_details,
            use_container_width=True
        )
