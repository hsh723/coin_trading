import pandas as pd
import numpy as np
from typing import Dict, Tuple

class AdvancedIndicators:
    @staticmethod
    def calculate_heikin_ashi(ohlc: pd.DataFrame) -> pd.DataFrame:
        """하이킨아시 캔들스틱 계산"""
        ha_close = (ohlc['open'] + ohlc['high'] + ohlc['low'] + ohlc['close']) / 4
        ha_open = pd.Series((ohlc['open'] + ohlc['close']).shift(1) / 2)
        ha_high = pd.Series(ohlc[['high', 'open', 'close']].max(axis=1))
        ha_low = pd.Series(ohlc[['low', 'open', 'close']].min(axis=1))
        
        return pd.DataFrame({
            'open': ha_open,
            'high': ha_high,
            'low': ha_low,
            'close': ha_close
        })

    @staticmethod
    def calculate_vwap(ohlc: pd.DataFrame, volume: pd.Series) -> pd.Series:
        """VWAP(Volume Weighted Average Price) 계산"""
        typical_price = (ohlc['high'] + ohlc['low'] + ohlc['close']) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
