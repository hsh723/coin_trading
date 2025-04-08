import numpy as np
from typing import Dict, List
from statsmodels.stats.diagnostic import het_white

class FractalAnalyzer:
    def __init__(self, min_window: int = 10, max_window: int = 100):
        self.min_window = min_window
        self.max_window = max_window
        
    def calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """허스트 지수 계산"""
        lags = range(self.min_window, self.max_window)
        tau = [np.sqrt(np.std(np.subtract(prices[lag:], prices[:-lag])))
               for lag in lags]
        
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0] / 2  # Hurst exponent
