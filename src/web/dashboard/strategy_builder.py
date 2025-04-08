import streamlit as st
import pandas as pd
from typing import Dict, List

class StrategyBuilderUI:
    def __init__(self):
        self.available_indicators = {
            'RSI': {'period': (5, 30, 14)},
            'MACD': {
                'fast_period': (5, 20, 12),
                'slow_period': (15, 40, 26),
                'signal_period': (5, 15, 9)
            },
            'Bollinger': {'window': (10, 30, 20), 'std': (1.5, 3.0, 2.0)}
        }
        
    def render(self):
        """전략 빌더 UI 렌더링"""
        st.header("Strategy Builder")
        
        with st.form("strategy_builder"):
            strategy_name = st.text_input("Strategy Name")
            selected_indicators = st.multiselect(
                "Select Technical Indicators",
                options=list(self.available_indicators.keys())
            )
            
            indicator_params = {}
            for indicator in selected_indicators:
                st.subheader(f"{indicator} Parameters")
                params = self._render_indicator_params(indicator)
                indicator_params[indicator] = params
