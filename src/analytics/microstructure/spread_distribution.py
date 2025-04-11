import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class SpreadDistribution:
    mean_spread: float
    spread_volatility: float
    spread_skewness: float
    spread_percentiles: Dict[str, float]

class SpreadDistributionAnalyzer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
    async def analyze_spread_distribution(self, market_data: Dict) -> SpreadDistribution:
        """스프레드 분포 분석"""
        spreads = self._calculate_spreads(market_data)
        
        return SpreadDistribution(
            mean_spread=np.mean(spreads),
            spread_volatility=np.std(spreads),
            spread_skewness=self._calculate_skewness(spreads),
            spread_percentiles=self._calculate_percentiles(spreads)
        )
