from typing import Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class MarketRegimeState:
    current_regime: str
    regime_probability: float
    transition_probability: Dict[str, float]
    regime_duration: int

class RealTimeRegimeAnalyzer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.regimes = ['low_volatility', 'normal', 'high_volatility', 'crisis']
        
    async def analyze_regime(self, market_data: pd.DataFrame) -> MarketRegimeState:
        """실시간 시장 국면 분석"""
        volatility = self._calculate_rolling_volatility(market_data)
        trend = self._calculate_trend_strength(market_data)
        
        current_regime = self._classify_regime(volatility, trend)
        regime_prob = self._calculate_regime_probability(volatility, trend)
        
        return MarketRegimeState(
            current_regime=current_regime,
            regime_probability=regime_prob,
            transition_probability=self._calculate_transition_probs(current_regime),
            regime_duration=self._calculate_regime_duration(current_regime)
        )
