from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class TrendAnalysis:
    trend_direction: str
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    breakdown_points: List[Dict]

class TrendAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'short_period': 10,
            'medium_period': 20,
            'long_period': 50
        }
        
    async def analyze_trend(self, price_data: np.ndarray) -> TrendAnalysis:
        """추세 분석"""
        direction = self._determine_trend_direction(price_data)
        strength = self._calculate_trend_strength(price_data)
        
        return TrendAnalysis(
            trend_direction=direction,
            trend_strength=strength,
            support_levels=self._find_support_levels(price_data),
            resistance_levels=self._find_resistance_levels(price_data),
            breakdown_points=self._identify_breakdown_points(price_data)
        )
