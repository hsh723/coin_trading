import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class MomentumMetrics:
    short_momentum: float
    medium_momentum: float
    long_momentum: float
    momentum_trend: str
    strength: float

class MomentumAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'short_period': 10,
            'medium_period': 30,
            'long_period': 90
        }
        
    def analyze_momentum(self, price_data: pd.Series) -> MomentumMetrics:
        """모멘텀 분석 수행"""
        short_mom = self._calculate_momentum(price_data, self.config['short_period'])
        medium_mom = self._calculate_momentum(price_data, self.config['medium_period'])
        long_mom = self._calculate_momentum(price_data, self.config['long_period'])
        
        return MomentumMetrics(
            short_momentum=short_mom,
            medium_momentum=medium_mom,
            long_momentum=long_mom,
            momentum_trend=self._determine_trend(short_mom, medium_mom, long_mom),
            strength=self._calculate_strength(short_mom, medium_mom, long_mom)
        )
