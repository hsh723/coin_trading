import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TimeWeightedMetrics:
    twap: float
    time_distribution: Dict[str, float]
    weighted_impact: float
    time_decay: List[float]

class TimeWeightedAnalyzer:
    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor
        
    async def analyze_time_weighted(self, market_data: Dict) -> TimeWeightedMetrics:
        """시간 가중 분석"""
        prices = np.array(market_data['prices'])
        volumes = np.array(market_data['volumes'])
        times = np.array(market_data['timestamps'])
        
        return TimeWeightedMetrics(
            twap=self._calculate_twap(prices, times),
            time_distribution=self._analyze_time_distribution(times),
            weighted_impact=self._calculate_weighted_impact(prices, volumes, times),
            time_decay=self._calculate_time_decay(times)
        )
