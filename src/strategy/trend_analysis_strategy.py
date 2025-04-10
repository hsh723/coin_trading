from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class TrendAnalysisResult:
    trend_direction: str  # 'up', 'down', 'sideways'
    trend_strength: float
    support_levels: List[float]
    resistance_levels: List[float]
    predicted_target: float

class TrendAnalysisStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'trend_period': 20,
            'strength_threshold': 0.6,
            'volume_weight': 0.3
        }
        
    async def analyze_trend(self, market_data: pd.DataFrame) -> TrendAnalysisResult:
        """추세 분석 실행"""
        ma_fast = self._calculate_moving_average(market_data, 20)
        ma_slow = self._calculate_moving_average(market_data, 50)
        volume_trend = self._analyze_volume_trend(market_data)
        
        trend_strength = self._calculate_trend_strength(ma_fast, ma_slow, volume_trend)
        support, resistance = self._find_support_resistance(market_data)
        
        return TrendAnalysisResult(
            trend_direction=self._determine_trend_direction(ma_fast, ma_slow),
            trend_strength=trend_strength,
            support_levels=support,
            resistance_levels=resistance,
            predicted_target=self._calculate_price_target(market_data, trend_strength)
        )
