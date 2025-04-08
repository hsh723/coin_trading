import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TapeReadingMetrics:
    buying_pressure: float
    selling_pressure: float
    large_orders: List[Dict]
    trade_flow: str
    
class TapeReader:
    def __init__(self, large_order_threshold: float = 10000):
        self.large_order_threshold = large_order_threshold
        
    def analyze_tape(self, trades: pd.DataFrame) -> TapeReadingMetrics:
        """시장 깊이 분석"""
        buy_pressure = self._calculate_buy_pressure(trades)
        sell_pressure = self._calculate_sell_pressure(trades)
        large_orders = self._identify_large_orders(trades)
        
        return TapeReadingMetrics(
            buying_pressure=buy_pressure,
            selling_pressure=sell_pressure,
            large_orders=large_orders,
            trade_flow=self._determine_flow(buy_pressure, sell_pressure)
        )
