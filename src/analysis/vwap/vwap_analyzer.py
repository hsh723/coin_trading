import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VWAPMetrics:
    current_vwap: float
    vwap_bands: Dict[str, float]
    price_deviation: float
    volume_profile: pd.Series

class VWAPAnalyzer:
    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        
    def calculate_vwap(self, data: pd.DataFrame) -> VWAPMetrics:
        """VWAP 및 관련 지표 계산"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        std = self._calculate_vwap_std(typical_price, data['volume'], vwap)
        
        return VWAPMetrics(
            current_vwap=vwap.iloc[-1],
            vwap_bands={
                'upper': vwap + 2 * std,
                'lower': vwap - 2 * std
            },
            price_deviation=(data['close'].iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1],
            volume_profile=self._calculate_volume_profile(data)
        )
