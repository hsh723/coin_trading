from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class RangeAnalysis:
    upper_bound: float
    lower_bound: float
    range_width: float
    range_strength: float
    breakout_probability: float

class RangeTradingStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_range_period': 20,
            'range_deviation': 0.02,
            'breakout_threshold': 0.03
        }
        
    async def analyze_range(self, market_data: pd.DataFrame) -> RangeAnalysis:
        """레인지 분석"""
        highs = market_data['high'].rolling(window=self.config['min_range_period']).max()
        lows = market_data['low'].rolling(window=self.config['min_range_period']).min()
        
        return RangeAnalysis(
            upper_bound=highs.iloc[-1],
            lower_bound=lows.iloc[-1],
            range_width=(highs.iloc[-1] - lows.iloc[-1]) / lows.iloc[-1],
            range_strength=self._calculate_range_strength(market_data),
            breakout_probability=self._calculate_breakout_probability(market_data)
        )
