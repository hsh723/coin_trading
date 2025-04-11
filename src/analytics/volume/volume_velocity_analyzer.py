import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolumeVelocity:
    velocity: float
    acceleration: float
    momentum_score: float
    velocity_trend: str

class VolumeVelocityAnalyzer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        
    async def analyze_velocity(self, volume_data: np.ndarray) -> VolumeVelocity:
        """거래량 속도 분석"""
        velocity = self._calculate_velocity(volume_data)
        acceleration = self._calculate_acceleration(velocity)
        
        return VolumeVelocity(
            velocity=velocity[-1],
            acceleration=acceleration[-1],
            momentum_score=self._calculate_momentum_score(velocity, acceleration),
            velocity_trend=self._determine_velocity_trend(velocity)
        )
