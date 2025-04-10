import streamlit as st
import pandas as pd
from dataclasses import dataclass

@dataclass
class AlertSummary:
    active_alerts: List[Dict]
    alert_history: pd.DataFrame
    severity_counts: Dict[str, int]
    response_times: Dict[str, float]

class AlertDashboard:
    def __init__(self):
        self.refresh_interval = 30  # 30초
        
    async def render_alert_dashboard(self, alert_data: Dict):
        """알림 대시보드 렌더링"""
        st.subheader("알림 모니터")
        
        summary = await self._generate_alert_summary(alert_data)
        
        # 알림 카운트
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="활성 알림",
                value=len(summary.active_alerts),
                delta=f"긴급 {summary.severity_counts.get('critical', 0)}건"
            )
            
        # 활성 알림 테이블
        st.write("활성 알림")
        if summary.active_alerts:
            alert_df = pd.DataFrame(summary.active_alerts)
            st.dataframe(alert_df)
