import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolatilityMetrics:
    current_volatility: float
    volatility_regime: str
    forecast_volatility: float
    regime_probabilities: Dict[str, float]

class MarketVolatilityAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'window_size': 30,
            'forecast_horizon': 5
        }
        
    async def analyze_volatility(self, market_data: pd.DataFrame) -> VolatilityMetrics:
        """변동성 분석 및 예측"""
        current_vol = self._calculate_current_volatility(market_data)
        forecast = self._forecast_volatility(market_data)
        regime = self._identify_volatility_regime(current_vol)
        
        return VolatilityMetrics(
            current_volatility=current_vol,
            volatility_regime=regime,
            forecast_volatility=forecast,
            regime_probabilities=self._calculate_regime_probabilities(market_data)
        )
