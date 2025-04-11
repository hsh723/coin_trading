import numpy as np
from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class OrderFlowMetrics:
    order_imbalance: float
    buy_pressure: float
    sell_pressure: float
    flow_strength: float

class OrderFlowAnalyzer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    async def analyze_flow(self, order_data: Dict) -> OrderFlowMetrics:
        """주문 흐름 분석"""
        buy_orders = self._calculate_buy_orders(order_data)
        sell_orders = self._calculate_sell_orders(order_data)
        total_flow = buy_orders + sell_orders
        
        return OrderFlowMetrics(
            order_imbalance=(buy_orders - sell_orders) / total_flow,
            buy_pressure=buy_orders / total_flow,
            sell_pressure=sell_orders / total_flow,
            flow_strength=self._calculate_flow_strength(order_data)
        )
