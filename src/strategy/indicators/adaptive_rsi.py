from typing import Dict
import numpy as np

class AdaptiveRSI:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'base_period': 14,
            'adaptive_factor': 0.1,
            'smoothing_factor': 2
        }
        
    async def calculate(self, price_data: np.ndarray) -> Dict[str, float]:
        """적응형 RSI 계산"""
        volatility = self._calculate_volatility(price_data)
        adjusted_period = self._adjust_period(volatility)
        
        return {
            'rsi': self._compute_rsi(price_data, adjusted_period),
            'adaptive_period': adjusted_period,
            'volatility_factor': volatility,
            'signal_strength': self._calculate_signal_strength(price_data)
        }
