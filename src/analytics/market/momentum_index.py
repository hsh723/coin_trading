import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class MomentumIndex:
    short_term: float
    medium_term: float
    long_term: float
    trend_strength: float
    momentum_score: float

class MomentumIndexAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'short_period': 10,
            'medium_period': 30,
            'long_period': 90
        }
        
    def calculate_momentum_index(self, price_data: pd.Series) -> MomentumIndex:
        """모멘텀 지수 계산"""
        return MomentumIndex(
            short_term=self._calculate_momentum(price_data, self.config['short_period']),
            medium_term=self._calculate_momentum(price_data, self.config['medium_period']),
            long_term=self._calculate_momentum(price_data, self.config['long_period']),
            trend_strength=self._calculate_trend_strength(price_data),
            momentum_score=self._calculate_momentum_score(price_data)
        )
