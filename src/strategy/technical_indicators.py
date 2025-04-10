from typing import Dict, List
import pandas as pd
import numpy as np

class TechnicalIndicators:
    def __init__(self):
        self.cache = {}
        
    async def calculate_indicators(self, price_data: pd.DataFrame) -> Dict:
        """기술적 지표 계산"""
        return {
            'rsi': self._calculate_rsi(price_data['close']),
            'macd': self._calculate_macd(price_data['close']),
            'bollinger': self._calculate_bollinger_bands(price_data['close']),
            'atr': self._calculate_atr(price_data),
            'momentum': self._calculate_momentum(price_data['close'])
        }
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
