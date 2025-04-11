import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolumeDivergence:
    price_trend: str
    volume_trend: str
    divergence_type: str
    divergence_strength: float

class VolumeDivergenceAnalyzer:
    def __init__(self, lookback_period: int = 14):
        self.lookback_period = lookback_period
        
    async def analyze_divergence(self, price_data: np.ndarray, volume_data: np.ndarray) -> VolumeDivergence:
        """거래량 다이버전스 분석"""
        price_trend = self._calculate_price_trend(price_data)
        volume_trend = self._calculate_volume_trend(volume_data)
        
        divergence_type = self._identify_divergence_type(price_trend, volume_trend)
        strength = self._calculate_divergence_strength(price_data, volume_data)
        
        return VolumeDivergence(
            price_trend=price_trend,
            volume_trend=volume_trend,
            divergence_type=divergence_type,
            divergence_strength=strength
        )
