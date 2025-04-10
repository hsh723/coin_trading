import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

@dataclass
class BacktestResults:
    performance_metrics: Dict[str, float]
    trade_history: pd.DataFrame
    equity_curve: pd.Series
    drawdown_periods: List[Dict]

class BacktestDashboard:
    def __init__(self):
        self.chart_height = 600
        
    async def render_backtest_results(self, results: BacktestResults):
        """백테스트 결과 대시보드 렌더링"""
        st.subheader("백테스트 결과 분석")
        
        # 주요 메트릭스
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="총 수익률",
                value=f"{results.performance_metrics['total_return']:.2f}%",
                delta=f"DD {results.performance_metrics['max_drawdown']:.1f}%"
            )
            
        # 수익률 곡선
        fig = self._create_equity_curve(results.equity_curve)
        st.plotly_chart(fig, use_container_width=True)
