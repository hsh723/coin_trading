import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class SizeDistributionMetrics:
    size_percentiles: Dict[str, float]
    distribution_skew: float
    large_trade_frequency: float
    size_clustering: Dict[str, List[float]]

class TradeSizeDistributionAnalyzer:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
    async def analyze_distribution(self, trades: List[Dict]) -> SizeDistributionMetrics:
        """거래 크기 분포 분석"""
        sizes = [trade['size'] for trade in trades[-self.window_size:]]
        percentiles = np.percentile(sizes, [25, 50, 75, 90, 95, 99])
        
        return SizeDistributionMetrics(
            size_percentiles=dict(zip(['p25', 'p50', 'p75', 'p90', 'p95', 'p99'], percentiles)),
            distribution_skew=self._calculate_distribution_skew(sizes),
            large_trade_frequency=self._calculate_large_trade_frequency(sizes),
            size_clustering=self._analyze_size_clusters(sizes)
        )
