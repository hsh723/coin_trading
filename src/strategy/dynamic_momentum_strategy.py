from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class DynamicMomentum:
    momentum_score: float
    trend_strength: float
    momentum_divergence: bool
    momentum_zones: List[Dict[str, float]]

class DynamicMomentumStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'momentum_periods': [14, 28, 56],
            'trend_threshold': 0.3,
            'divergence_lookback': 20
        }
        
    async def analyze_momentum(self, market_data: pd.DataFrame) -> DynamicMomentum:
        """동적 모멘텀 분석"""
        momentum_values = self._calculate_multi_period_momentum(market_data)
        trend = self._analyze_momentum_trend(momentum_values)
        
        return DynamicMomentum(
            momentum_score=self._calculate_composite_momentum(momentum_values),
            trend_strength=trend['strength'],
            momentum_divergence=self._check_momentum_divergence(market_data, momentum_values),
            momentum_zones=self._identify_momentum_zones(momentum_values)
        )
