from typing import Dict, List
import numpy as np
from scipy.stats import norm

class MarketRegimeDetector:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'window_size': 60,
            'regime_types': ['trending', 'mean_reverting', 'volatility']
        }
        
    async def detect_regime(self, price_data: np.ndarray) -> Dict:
        """현재 시장 상태 감지"""
        volatility = self._calculate_rolling_volatility(price_data)
        trend = self._calculate_trend_strength(price_data)
        mean_reversion = self._calculate_mean_reversion(price_data)
        
        return {
            'current_regime': self._classify_regime(volatility, trend, mean_reversion),
            'regime_metrics': {
                'volatility': float(volatility),
                'trend_strength': float(trend),
                'mean_reversion': float(mean_reversion)
            },
            'confidence': self._calculate_regime_confidence()
        }
