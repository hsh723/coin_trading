import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class IntradayMetrics:
    volume_profile: Dict[str, float]
    price_discovery: List[float]
    trade_intensity: float
    intraday_pattern: str

class IntradayAnalyzer:
    def __init__(self, intervals: int = 24):
        self.intervals = intervals
        
    async def analyze_intraday(self, market_data: Dict) -> IntradayMetrics:
        """일중 패턴 분석"""
        hourly_data = self._aggregate_hourly_data(market_data)
        discovery = self._analyze_price_discovery(hourly_data)
        
        return IntradayMetrics(
            volume_profile=self._calculate_volume_profile(hourly_data),
            price_discovery=discovery,
            trade_intensity=self._calculate_trade_intensity(hourly_data),
            intraday_pattern=self._identify_pattern(hourly_data)
        )
