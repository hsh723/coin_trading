import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats

class MarketEfficiencyAnalyzer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
    def calculate_efficiency_ratio(self, data: pd.DataFrame) -> float:
        """시장 효율성 비율 계산"""
        price_changes = data['close'].diff().abs().sum()
        direct_movement = abs(data['close'].iloc[-1] - data['close'].iloc[0])
        return direct_movement / price_changes if price_changes != 0 else 0

    def analyze_market_efficiency(self, data: pd.DataFrame) -> Dict:
        """시장 효율성 분석"""
        returns = np.log(data['close']).diff()
        
        return {
            'efficiency_ratio': self.calculate_efficiency_ratio(data),
            'hurst_exponent': self._calculate_hurst_exponent(returns),
            'random_walk_test': self._perform_random_walk_test(returns)
        }
