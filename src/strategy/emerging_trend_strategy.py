from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class EmergingTrend:
    trend_type: str  # emerging, confirming, exhausting
    trend_age: int
    momentum_score: float
    confirmation_signals: List[str]
    risk_level: float

class EmergingTrendStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'detection_window': 20,
            'confirmation_threshold': 0.7,
            'momentum_periods': [5, 10, 20]
        }
        
    async def detect_emerging_trends(self, market_data: pd.DataFrame) -> EmergingTrend:
        """이머징 트렌드 감지"""
        momentum_scores = self._calculate_momentum_scores(market_data)
        trend_type = self._classify_trend_type(momentum_scores)
        
        return EmergingTrend(
            trend_type=trend_type,
            trend_age=self._calculate_trend_age(market_data),
            momentum_score=momentum_scores['combined'],
            confirmation_signals=self._get_confirmation_signals(market_data),
            risk_level=self._assess_trend_risk(trend_type, momentum_scores)
        )
