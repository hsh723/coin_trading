from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class ContinuationIndex:
    trend_strength: float
    continuation_probability: float
    reversal_zones: List[float]
    momentum_alignment: bool

class ContinuationIndexStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'lookback_period': 14,
            'momentum_threshold': 0.6,
            'min_trend_strength': 0.4
        }
        
    async def analyze_continuation(self, market_data: pd.DataFrame) -> ContinuationIndex:
        """연속성 지수 분석"""
        trend_strength = self._calculate_trend_strength(market_data)
        momentum = self._calculate_momentum(market_data)
        
        return ContinuationIndex(
            trend_strength=trend_strength,
            continuation_probability=self._calculate_probability(trend_strength, momentum),
            reversal_zones=self._identify_reversal_zones(market_data),
            momentum_alignment=momentum > self.config['momentum_threshold']
        )
