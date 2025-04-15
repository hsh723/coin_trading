import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class MomentumSignal:
    momentum_score: float
    trend_strength: float
    momentum_regime: str
    reversal_probability: float

class MomentumAnalyzer:
    def __init__(self, window_size: int = 14):
        self.window_size = window_size
        
    async def analyze_momentum(self, price_data: np.ndarray) -> MomentumSignal:
        """모멘텀 분석"""
        score = self._calculate_momentum_score(price_data)
        strength = self._calculate_trend_strength(price_data)
        
        return MomentumSignal(
            momentum_score=score,
            trend_strength=strength,
            momentum_regime=self._identify_momentum_regime(score, strength),
            reversal_probability=self._calculate_reversal_probability(price_data)
        )
