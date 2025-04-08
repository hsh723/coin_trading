import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class HeikinAshiMetrics:
    trend_strength: float
    trend_direction: str
    reversal_signals: List[Dict]
    pattern_confidence: float

class HeikinAshiAnalyzer:
    def __init__(self, smoothing_period: int = 10):
        self.smoothing_period = smoothing_period
        
    def calculate_heikin_ashi(self, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """헤이킨 아시 캔들 계산"""
        ha_close = (ohlc_data['open'] + ohlc_data['high'] +
                   ohlc_data['low'] + ohlc_data['close']) / 4
        ha_open = pd.Series((ohlc_data['open'] + ohlc_data['close']).shift(1) / 2)
        ha_high = pd.Series(ohlc_data[['high', 'open', 'close']].max(axis=1))
        ha_low = pd.Series(ohlc_data[['low', 'open', 'close']].min(axis=1))
        
        return pd.DataFrame({
            'open': ha_open,
            'high': ha_high,
            'low': ha_low,
            'close': ha_close
        })
