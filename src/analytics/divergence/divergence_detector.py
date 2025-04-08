import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Divergence:
    type: str  # 'regular' or 'hidden'
    direction: str  # 'bullish' or 'bearish'
    start_index: int
    end_index: int
    strength: float

class DivergenceDetector:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_swing_size': 0.02,
            'lookback_period': 50
        }
        
    def detect_divergences(self, price_data: pd.Series, 
                         indicator_data: pd.Series) -> List[Divergence]:
        """RSI/MACD 다이버전스 감지"""
        price_swings = self._find_swing_points(price_data)
        indicator_swings = self._find_swing_points(indicator_data)
        
        divergences = []
        for i in range(len(price_swings) - 1):
            if self._is_divergence(price_swings[i:i+2], 
                                 indicator_swings[i:i+2]):
                divergences.append(
                    self._create_divergence(price_swings[i:i+2], 
                                          indicator_swings[i:i+2])
                )
        return divergences
