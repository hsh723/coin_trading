import numpy as np
from typing import Dict, List

class VolumeVolatilityAnalyzer:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.volatility_history = []
        
    async def analyze_volatility(self, volume_data: np.ndarray) -> Dict:
        """거래량 변동성 분석"""
        volatility = np.std(volume_data) / np.mean(volume_data)
        self.volatility_history.append(volatility)
        
        return {
            'current_volatility': volatility,
            'historical_trend': self._analyze_volatility_trend(),
            'volatility_regime': self._detect_volatility_regime(),
            'abnormal_levels': self._detect_abnormal_levels()
        }
