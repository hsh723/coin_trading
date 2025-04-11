import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolumeCorrelation:
    price_volume_correlation: float
    lead_lag_correlation: Dict[str, float]
    correlation_trend: str
    significance_level: float

class VolumeCorrelationAnalyzer:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        
    async def analyze_correlation(self, price_data: np.ndarray, volume_data: np.ndarray) -> VolumeCorrelation:
        """거래량-가격 상관관계 분석"""
        price_vol_corr = np.corrcoef(price_data, volume_data)[0, 1]
        lead_lag = self._calculate_lead_lag_correlation(price_data, volume_data)
        
        return VolumeCorrelation(
            price_volume_correlation=price_vol_corr,
            lead_lag_correlation=lead_lag,
            correlation_trend=self._determine_correlation_trend(price_vol_corr),
            significance_level=self._calculate_significance(price_vol_corr)
        )
