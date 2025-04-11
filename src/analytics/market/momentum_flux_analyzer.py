import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MomentumFlux:
    momentum_score: float
    flux_direction: str
    momentum_regime: str
    acceleration: float

class MomentumFluxAnalyzer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        
    async def analyze_momentum_flux(self, price_data: np.ndarray, volume_data: np.ndarray) -> MomentumFlux:
        """모멘텀 변화 분석"""
        momentum = self._calculate_momentum(price_data)
        flux = self._calculate_flux(momentum)
        regime = self._identify_regime(momentum, flux)
        
        return MomentumFlux(
            momentum_score=momentum[-1],
            flux_direction=self._determine_direction(flux),
            momentum_regime=regime,
            acceleration=self._calculate_acceleration(momentum)
        )
