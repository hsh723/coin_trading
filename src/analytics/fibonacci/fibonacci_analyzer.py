import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class FibonacciLevels:
    retracement_levels: Dict[float, float]
    extension_levels: Dict[float, float]
    time_zones: List[pd.Timestamp]

class FibonacciAnalyzer:
    def __init__(self):
        self.retracement_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.extension_ratios = [1.272, 1.618, 2.618]
        
    def calculate_levels(self, high: float, low: float) -> FibonacciLevels:
        """피보나치 레벨 계산"""
        diff = high - low
        retracements = {
            ratio: high - (diff * ratio)
            for ratio in self.retracement_ratios
        }
        
        extensions = {
            ratio: high + (diff * ratio)
            for ratio in self.extension_ratios
        }
        
        return FibonacciLevels(
            retracement_levels=retracements,
            extension_levels=extensions,
            time_zones=self._calculate_time_zones()
        )
