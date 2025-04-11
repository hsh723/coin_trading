import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolumeMomentum:
    momentum_score: float
    momentum_direction: str
    acceleration: float
    breakout_probability: float

class VolumeMomentumAnalyzer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.momentum_history = []
        
    async def analyze_momentum(self, volume_data: np.ndarray) -> VolumeMomentum:
        """거래량 모멘텀 분석"""
        momentum = self._calculate_momentum(volume_data)
        acceleration = self._calculate_acceleration(momentum)
        
        return VolumeMomentum(
            momentum_score=momentum[-1],
            momentum_direction=self._determine_direction(momentum),
            acceleration=acceleration,
            breakout_probability=self._estimate_breakout_probability(momentum)
        )
