import numpy as np
import pandas as pd
from typing import List, Dict
from scipy.signal import argrelextrema

class SupportResistanceDetector:
    def __init__(self, window_size: int = 20, threshold: float = 0.02):
        self.window_size = window_size
        self.threshold = threshold
        
    def detect_levels(self, price_data: pd.Series) -> Dict[str, List[float]]:
        """지지/저항 레벨 감지"""
        highs = self._find_local_extrema(price_data, np.greater)
        lows = self._find_local_extrema(price_data, np.less)
        
        return {
            'support_levels': self._cluster_levels(lows),
            'resistance_levels': self._cluster_levels(highs),
            'current_zone': self._identify_current_zone(price_data.iloc[-1], 
                                                      lows, highs)
        }
