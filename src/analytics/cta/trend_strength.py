import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class TrendStrength:
    strength: float
    direction: str
    momentum: float
    persistence: float

class TrendStrengthAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'short_period': 20,
            'long_period': 50,
            'momentum_period': 14
        }
        
    def analyze_trend(self, price_data: pd.Series) -> TrendStrength:
        """추세 강도 분석"""
        direction = self._determine_trend_direction(price_data)
        strength = self._calculate_trend_strength(price_data)
        momentum = self._calculate_momentum(price_data)
        persistence = self._calculate_trend_persistence(price_data)
        
        return TrendStrength(
            strength=strength,
            direction=direction,
            momentum=momentum,
            persistence=persistence
        )
