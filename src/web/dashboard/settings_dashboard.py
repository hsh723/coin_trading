import streamlit as st
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class SystemSettings:
    trading_params: Dict[str, Any]
    risk_limits: Dict[str, float]
    notification_config: Dict[str, bool]

class SettingsDashboard:
    def __init__(self):
        self.settings_path = "config/settings.yaml"
        
    async def render_settings_dashboard(self, current_settings: SystemSettings):
        """설정 대시보드 렌더링"""
        st.subheader("시스템 설정")
        
        # 트레이딩 파라미터 설정
        st.write("트레이딩 파라미터")
        col1, col2 = st.columns(2)
        with col1:
            leverage = st.slider("레버리지", 1, 100, 
                               current_settings.trading_params.get('leverage', 1))
            position_size = st.slider("포지션 크기 (%)", 1, 100,
                                    current_settings.trading_params.get('position_size', 10))
                                    
        # 리스크 설정
        st.write("리스크 관리")
        stop_loss = st.slider("손절 비율 (%)", 0.1, 10.0,
                            current_settings.risk_limits.get('stop_loss', 2.0))
