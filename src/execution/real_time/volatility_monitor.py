import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class VolatilityState:
    current_volatility: float
    regime: str
    forecast: float
    alert_level: str

class VolatilityMonitor:
    def __init__(self, alert_threshold: float = 0.02):
        self.alert_threshold = alert_threshold
        self.volatility_window = []
        
    async def monitor_volatility(self, price_data: np.ndarray) -> VolatilityState:
        """실시간 변동성 모니터링"""
        current_vol = self._calculate_current_volatility(price_data)
        regime = self._determine_volatility_regime(current_vol)
        
        return VolatilityState(
            current_volatility=current_vol,
            regime=regime,
            forecast=self._forecast_volatility(),
            alert_level=self._determine_alert_level(current_vol)
        )
