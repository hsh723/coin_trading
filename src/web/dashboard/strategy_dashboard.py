import streamlit as st
import pandas as pd
from dataclasses import dataclass

@dataclass
class StrategyMetrics:
    active_strategies: List[Dict]
    strategy_performance: pd.DataFrame
    risk_metrics: Dict[str, float]
    optimization_status: Dict[str, any]

class StrategyDashboard:
    def __init__(self):
        self.refresh_interval = 60  # 1분
        
    async def render_strategy_dashboard(self, strategy_data: StrategyMetrics):
        """전략 모니터링 대시보드 렌더링"""
        st.subheader("전략 모니터링")
        
        # 활성 전략 목록
        st.write("활성 전략")
        active_df = pd.DataFrame(strategy_data.active_strategies)
        st.dataframe(active_df)
        
        # 전략별 성과
        st.write("전략별 성과")
        st.line_chart(strategy_data.strategy_performance)
