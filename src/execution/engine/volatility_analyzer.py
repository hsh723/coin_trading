from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class VolatilityMetrics:
    current_volatility: float
    volatility_trend: str
    expected_range: Dict[str, float]
    risk_level: str
    regime_state: str

class VolatilityAnalyzer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        
    async def analyze_volatility(self, market_data: Dict) -> VolatilityMetrics:
        """변동성 분석"""
        prices = np.array(market_data['prices'])
        returns = np.diff(np.log(prices))
        
        volatility = self._calculate_volatility(returns)
        trend = self._analyze_volatility_trend(returns)
        
        return VolatilityMetrics(
            current_volatility=volatility,
            volatility_trend=trend,
            expected_range=self._calculate_expected_range(volatility),
            risk_level=self._determine_risk_level(volatility),
            regime_state=self._detect_regime(volatility, trend)
        )
