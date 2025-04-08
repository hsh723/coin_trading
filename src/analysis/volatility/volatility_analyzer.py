import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats

class VolatilityAnalyzer:
    def __init__(self, window_sizes: List[int] = [14, 30, 90]):
        self.window_sizes = window_sizes
        
    def analyze_volatility_regime(self, returns: pd.Series) -> Dict[str, float]:
        """변동성 국면 분석"""
        current_vol = returns.rolling(window=self.window_sizes[0]).std() * np.sqrt(252)
        historical_vol = returns.rolling(window=self.window_sizes[-1]).std() * np.sqrt(252)
        
        vol_ratio = current_vol.iloc[-1] / historical_vol.iloc[-1]
        
        if vol_ratio > 1.5:
            regime = "HIGH_VOLATILITY"
        elif vol_ratio < 0.75:
            regime = "LOW_VOLATILITY"
        else:
            regime = "NORMAL_VOLATILITY"
            
        return {
            'regime': regime,
            'current_vol': current_vol.iloc[-1],
            'historical_vol': historical_vol.iloc[-1],
            'vol_ratio': vol_ratio
        }
