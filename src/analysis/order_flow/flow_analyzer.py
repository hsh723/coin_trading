import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class OrderFlowMetrics:
    buy_volume: float
    sell_volume: float
    net_flow: float
    pressure_index: float

class OrderFlowAnalyzer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        
    def analyze_flow(self, trades: pd.DataFrame) -> OrderFlowMetrics:
        """주문흐름 분석"""
        buy_trades = trades[trades['side'] == 'buy']
        sell_trades = trades[trades['side'] == 'sell']
        
        metrics = OrderFlowMetrics(
            buy_volume=buy_trades['volume'].sum(),
            sell_volume=sell_trades['volume'].sum(),
            net_flow=buy_trades['volume'].sum() - sell_trades['volume'].sum(),
            pressure_index=self._calculate_pressure_index(trades)
        )
        return metrics
