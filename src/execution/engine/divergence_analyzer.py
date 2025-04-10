from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class DivergenceSignal:
    type: str  # regular, hidden
    strength: float
    price_points: List[float]
    indicator_points: List[float]
    confidence: float

class DivergenceAnalyzer:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
        
    async def analyze_divergence(self, 
                               price_data: np.ndarray, 
                               indicator_data: np.ndarray) -> DivergenceSignal:
        """다이버전스 분석"""
        price_extremes = self._find_extremes(price_data)
        indicator_extremes = self._find_extremes(indicator_data)
        
        return DivergenceSignal(
            type=self._determine_divergence_type(price_extremes, indicator_extremes),
            strength=self._calculate_divergence_strength(price_extremes, indicator_extremes),
            price_points=price_extremes,
            indicator_points=indicator_extremes,
            confidence=self._calculate_confidence(price_extremes, indicator_extremes)
        )
