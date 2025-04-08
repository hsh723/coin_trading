import streamlit as st
import plotly.graph_objects as go
from typing import Dict
import pandas as pd

class OrderBookView:
    def __init__(self, depth: int = 20):
        self.depth = depth
        self.colors = {
            'bid': 'rgba(0, 255, 0, 0.5)',
            'ask': 'rgba(255, 0, 0, 0.5)'
        }
        
    def render(self, order_book: Dict[str, pd.DataFrame]):
        """오더북 시각화"""
        fig = go.Figure()
        
        # 매수 주문
        fig.add_trace(go.Bar(
            x=order_book['bids']['price'],
            y=order_book['bids']['volume'],
            name='Bids',
            marker_color=self.colors['bid']
        ))
        
        # 매도 주문
        fig.add_trace(go.Bar(
            x=order_book['asks']['price'],
            y=order_book['asks']['volume'],
            name='Asks',
            marker_color=self.colors['ask']
        ))
        
        st.plotly_chart(fig)
