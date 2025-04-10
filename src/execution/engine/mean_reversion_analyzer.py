from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class ReversionMetrics:
    zscore: float
    mean_level: float
    deviation_level: float
    reversion_probability: float
    target_levels: List[float]

class MeanReversionAnalyzer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        
    async def analyze_reversion(self, price_data: np.ndarray) -> ReversionMetrics:
        """평균 회귀 분석"""
        mean = np.mean(price_data[-self.window_size:])
        std = np.std(price_data[-self.window_size:])
        current_price = price_data[-1]
        
        zscore = (current_price - mean) / std
        
        return ReversionMetrics(
            zscore=zscore,
            mean_level=mean,
            deviation_level=abs(current_price - mean) / mean,
            reversion_probability=self._calculate_reversion_probability(zscore),
            target_levels=self._calculate_target_levels(mean, std)
        )
