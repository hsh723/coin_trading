from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class MarketEfficiency:
    hurst_exponent: float
    efficiency_ratio: float
    randomness_score: float
    predictability: float

class MarketEfficiencyAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'window_size': 100,
            'min_samples': 50,
            'significance_level': 0.05
        }
        
    async def analyze_efficiency(self, price_data: np.ndarray) -> MarketEfficiency:
        """시장 효율성 분석"""
        hurst = self._calculate_hurst_exponent(price_data)
        efficiency = self._calculate_efficiency_ratio(price_data)
        
        return MarketEfficiency(
            hurst_exponent=hurst,
            efficiency_ratio=efficiency,
            randomness_score=self._calculate_randomness(price_data),
            predictability=1 - efficiency
        )
