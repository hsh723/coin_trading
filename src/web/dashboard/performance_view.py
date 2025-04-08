import streamlit as st
import plotly.graph_objects as go
from typing import Dict
import pandas as pd

class PerformanceView:
    def __init__(self):
        self.metrics_cache = {}
        
    def render(self, performance_data: Dict):
        """성능 대시보드 렌더링"""
        st.header("Performance Analytics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            self._render_returns_metrics(performance_data)
        with col2:
            self._render_risk_metrics(performance_data)
        with col3:
            self._render_trading_metrics(performance_data)
            
        self._plot_equity_curve(performance_data)
