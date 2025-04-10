from typing import Dict, List
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class MomentumSignals:
    rsi_signal: float
    macd_signal: Dict[str, float]
    momentum_score: float
    trend_strength: float

class MomentumIndicator:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }
        
    async def calculate_momentum(self, price_data: pd.Series) -> MomentumSignals:
        """모멘텀 지표 계산"""
        rsi = self._calculate_rsi(price_data)
        macd = self._calculate_macd(price_data)
        momentum = self._calculate_momentum_score(price_data)
        
        return MomentumSignals(
            rsi_signal=rsi[-1],
            macd_signal=macd,
            momentum_score=momentum,
            trend_strength=self._calculate_trend_strength(price_data)
        )
