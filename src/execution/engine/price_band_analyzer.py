from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class PriceBandMetrics:
    upper_band: float
    lower_band: float
    middle_band: float
    band_width: float
    position_within_band: float

class PriceBandAnalyzer:
    def __init__(self, window_size: int = 20, num_std: float = 2.0):
        self.window_size = window_size
        self.num_std = num_std
        
    async def analyze_price_bands(self, price_data: np.ndarray) -> PriceBandMetrics:
        """가격 밴드 분석"""
        rolling_mean = np.mean(price_data[-self.window_size:])
        rolling_std = np.std(price_data[-self.window_size:])
        
        upper = rolling_mean + (self.num_std * rolling_std)
        lower = rolling_mean - (self.num_std * rolling_std)
        current_price = price_data[-1]
        
        return PriceBandMetrics(
            upper_band=upper,
            lower_band=lower,
            middle_band=rolling_mean,
            band_width=(upper - lower) / rolling_mean,
            position_within_band=self._calculate_position(current_price, upper, lower)
        )
