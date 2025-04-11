from typing import Dict, List
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'time_features': ['hour', 'day', 'month'],
            'technical_features': ['rsi', 'macd', 'bbands'],
            'volume_features': ['vwap', 'obv']
        }
        
    async def create_features(self, market_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """특징 엔지니어링"""
        return {
            'time_based': self._create_time_features(market_data),
            'technical': self._create_technical_features(market_data),
            'volume_based': self._create_volume_features(market_data),
            'interaction': self._create_interaction_features(market_data)
        }
