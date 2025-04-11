import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class WeightedVolumeMetrics:
    time_weighted_volume: float
    price_weighted_volume: float
    volume_power: float
    buying_pressure: float

class WeightedVolumeAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'time_decay': 0.95,
            'volume_threshold': 1.5
        }
        
    async def analyze_weighted_volume(self, market_data: Dict) -> WeightedVolumeMetrics:
        """가중 거래량 분석"""
        twv = self._calculate_time_weighted_volume(market_data)
        pwv = self._calculate_price_weighted_volume(market_data)
        
        return WeightedVolumeMetrics(
            time_weighted_volume=twv,
            price_weighted_volume=pwv,
            volume_power=self._calculate_volume_power(twv, pwv),
            buying_pressure=self._calculate_buying_pressure(market_data)
        )
