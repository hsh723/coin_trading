from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class TimeframeSignal:
    selected_timeframe: str
    volatility_score: float
    timeframe_weights: Dict[str, float]
    optimal_indicators: List[str]

class AdaptiveTimeframeStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'timeframes': ['1m', '5m', '15m', '1h', '4h'],
            'volatility_window': 24,
            'min_data_points': 100
        }
        
    async def select_timeframe(self, market_data: Dict[str, pd.DataFrame]) -> TimeframeSignal:
        """최적 타임프레임 선택"""
        volatilities = {
            tf: self._calculate_volatility(data) 
            for tf, data in market_data.items()
        }
        
        weights = self._calculate_timeframe_weights(volatilities)
        selected = max(weights.items(), key=lambda x: x[1])[0]
        
        return TimeframeSignal(
            selected_timeframe=selected,
            volatility_score=volatilities[selected],
            timeframe_weights=weights,
            optimal_indicators=self._select_indicators(selected)
        )
