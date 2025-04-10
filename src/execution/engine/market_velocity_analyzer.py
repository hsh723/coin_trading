from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class MarketVelocity:
    price_velocity: float
    volume_velocity: float
    momentum_score: float
    acceleration: float
    trend_strength: float

class MarketVelocityAnalyzer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        
    async def analyze_velocity(self, market_data: Dict) -> MarketVelocity:
        """시장 속도 분석"""
        price_data = np.array(market_data['prices'])
        volume_data = np.array(market_data['volumes'])
        
        return MarketVelocity(
            price_velocity=self._calculate_price_velocity(price_data),
            volume_velocity=self._calculate_volume_velocity(volume_data),
            momentum_score=self._calculate_momentum(price_data),
            acceleration=self._calculate_acceleration(price_data),
            trend_strength=self._calculate_trend_strength(price_data)
        )
