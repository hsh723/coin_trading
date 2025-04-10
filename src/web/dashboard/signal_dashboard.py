import streamlit as st
import pandas as pd
from dataclasses import dataclass

@dataclass
class SignalData:
    active_signals: List[Dict]
    signal_history: pd.DataFrame
    signal_performance: Dict[str, float]
    signal_distribution: Dict[str, int]

class SignalDashboard:
    def __init__(self):
        self.refresh_interval = 30  # 30초
        
    async def render_signal_dashboard(self, signal_data: SignalData):
        """신호 대시보드 렌더링"""
        st.subheader("신호 모니터링")
        
        # 활성 신호
        st.write("활성 신호")
        active_signals_df = pd.DataFrame(signal_data.active_signals)
        st.dataframe(active_signals_df)
        
        # 신호 성과
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="신호 정확도",
                value=f"{signal_data.signal_performance['accuracy']:.1f}%"
            )
