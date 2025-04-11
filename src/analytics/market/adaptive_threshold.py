import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class AdaptiveThresholds:
    upper_threshold: float
    lower_threshold: float
    volatility_adjusted: bool
    threshold_breaches: List[Dict]

class AdaptiveThresholdAnalyzer:
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        
    async def calculate_thresholds(self, market_data: pd.DataFrame) -> AdaptiveThresholds:
        """적응형 임계값 계산"""
        volatility = self._calculate_volatility(market_data)
        upper = self._calculate_upper_threshold(market_data, volatility)
        lower = self._calculate_lower_threshold(market_data, volatility)
        
        return AdaptiveThresholds(
            upper_threshold=upper,
            lower_threshold=lower,
            volatility_adjusted=True,
            threshold_breaches=self._detect_breaches(market_data, upper, lower)
        )
