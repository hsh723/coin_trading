import streamlit as st
import plotly.graph_objects as go
from typing import Dict, List

class StrategyMonitor:
    def __init__(self):
        self.metrics_cache = {}
        
    def render_dashboard(self, strategy_results: Dict):
        """전략 모니터링 대시보드 렌더링"""
        st.title("Strategy Performance Monitor")
        
        # Performance Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            self._render_returns_metric(strategy_results)
        with col2:
            self._render_risk_metrics(strategy_results)
        with col3:
            self._render_trade_metrics(strategy_results)
