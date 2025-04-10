import streamlit as st
from dataclasses import dataclass

@dataclass
class DashboardConfig:
    title: str
    refresh_rate: int
    layout: Dict[str, Dict]
    theme: Dict[str, str]

class MainDashboard:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'title': 'Crypto Trading Dashboard',
            'refresh_rate': 5,
            'theme': 'light'
        }
        
    async def render_dashboard(self):
        """대시보드 렌더링"""
        st.set_page_config(
            page_title=self.config['title'],
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 헤더 섹션
        st.title(self.config['title'])
        
        # 메인 레이아웃
        col1, col2, col3 = st.columns(3)
        
        with col1:
            await self._render_market_overview()
            
        with col2:
            await self._render_portfolio_status()
            
        with col3:
            await self._render_trading_stats()
