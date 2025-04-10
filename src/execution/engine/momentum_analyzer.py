from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class MomentumMetrics:
    short_momentum: float
    medium_momentum: float
    long_momentum: float
    momentum_signal: str
    strength: float

class MomentumAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'short_period': 5,
            'medium_period': 15,
            'long_period': 30
        }
        
    async def analyze_momentum(self, price_data: np.ndarray) -> MomentumMetrics:
        """모멘텀 분석"""
        short_mom = self._calculate_momentum(price_data, self.config['short_period'])
        medium_mom = self._calculate_momentum(price_data, self.config['medium_period'])
        long_mom = self._calculate_momentum(price_data, self.config['long_period'])
        
        signal = self._generate_momentum_signal(short_mom, medium_mom, long_mom)
        strength = self._calculate_momentum_strength(short_mom, medium_mom, long_mom)
        
        return MomentumMetrics(
            short_momentum=short_mom,
            medium_momentum=medium_mom,
            long_momentum=long_mom,
            momentum_signal=signal,
            strength=strength
        )
