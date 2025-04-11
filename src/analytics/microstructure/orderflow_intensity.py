import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class OrderFlowIntensity:
    intensity_score: float
    flow_imbalance: float
    order_arrival_rate: float
    market_pressure: str

class OrderFlowIntensityAnalyzer:
    def __init__(self, intensity_window: int = 50):
        self.intensity_window = intensity_window
        
    async def analyze_intensity(self, order_data: Dict) -> OrderFlowIntensity:
        """주문 흐름 강도 분석"""
        intensity = self._calculate_intensity(order_data)
        imbalance = self._calculate_imbalance(order_data)
        
        return OrderFlowIntensity(
            intensity_score=intensity,
            flow_imbalance=imbalance,
            order_arrival_rate=self._calculate_arrival_rate(order_data),
            market_pressure='buy' if imbalance > 0 else 'sell'
        )
