import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MarketPressure:
    buy_pressure: float
    sell_pressure: float
    net_pressure: float
    pressure_zones: List[Dict[str, float]]

class MarketPressureAnalyzer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    async def analyze_pressure(self, order_book: Dict, trades: List[Dict]) -> MarketPressure:
        """시장 압력 분석"""
        buy_pressure = self._calculate_buy_pressure(order_book, trades)
        sell_pressure = self._calculate_sell_pressure(order_book, trades)
        
        return MarketPressure(
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            net_pressure=buy_pressure - sell_pressure,
            pressure_zones=self._identify_pressure_zones(order_book)
        )
