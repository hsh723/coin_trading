from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class PivotPoints:
    pivot: float
    support_levels: List[float]
    resistance_levels: List[float]
    breakout_status: str

class PivotPointStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'method': 'standard',  # standard, fibonacci, woodie
            'levels': 3,  # number of S/R levels
            'breakout_threshold': 0.002  # 0.2%
        }
        
    async def calculate_pivot_points(self, market_data: pd.DataFrame) -> PivotPoints:
        """피벗 포인트 계산"""
        high = market_data['high'].iloc[-1]
        low = market_data['low'].iloc[-1]
        close = market_data['close'].iloc[-1]
        
        pivot = (high + low + close) / 3
        support_levels = self._calculate_support_levels(pivot, high, low)
        resistance_levels = self._calculate_resistance_levels(pivot, high, low)
        
        return PivotPoints(
            pivot=pivot,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            breakout_status=self._check_breakout_status(close, support_levels, resistance_levels)
        )
