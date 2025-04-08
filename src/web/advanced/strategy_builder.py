import streamlit as st
from typing import Dict, List

class StrategyBuilder:
    def __init__(self):
        self.available_indicators = {
            'RSI': {'period': 14},
            'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
            'Bollinger': {'period': 20, 'std': 2}
        }
        
    def render(self):
        """전략 빌더 UI 렌더링"""
        st.header("Strategy Builder")
        
        # 지표 선택
        selected_indicators = st.multiselect(
            "Select Indicators",
            list(self.available_indicators.keys())
        )
        
        # 지표 파라미터 설정
        strategy_config = self._render_indicator_params(selected_indicators)
        
        # 규칙 설정
        rules = self._render_rule_builder(selected_indicators)
        
        return {
            'indicators': strategy_config,
            'rules': rules
        }
