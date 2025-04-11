import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TrendStrength:
    strength_score: float
    trend_direction: str
    trend_reliability: float
    breakout_signals: List[Dict]

class TrendStrengthAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'strength_window': 20,
            'breakout_threshold': 2.0
        }
        
    async def analyze_trend_strength(self, market_data: pd.DataFrame) -> TrendStrength:
        """추세 강도 분석"""
        score = self._calculate_strength_score(market_data)
        direction = self._determine_trend_direction(market_data)
        
        return TrendStrength(
            strength_score=score,
            trend_direction=direction,
            trend_reliability=self._calculate_reliability(market_data),
            breakout_signals=self._detect_breakout_signals(market_data)
        )
