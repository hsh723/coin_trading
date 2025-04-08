import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RenkoMetrics:
    brick_size: float
    trend_direction: str
    brick_count: int
    reversal_points: List[Dict]

class RenkoAnalyzer:
    def __init__(self, brick_size: float = None, atr_period: int = 14):
        self.brick_size = brick_size
        self.atr_period = atr_period
        
    def build_renko(self, price_data: pd.Series) -> Dict:
        """렌코 차트 구축 및 분석"""
        if not self.brick_size:
            self.brick_size = self._calculate_optimal_brick_size(price_data)
            
        renko_data = self._construct_renko_series(price_data)
        trend = self._analyze_trend(renko_data)
        
        return {
            'renko_data': renko_data,
            'metrics': RenkoMetrics(
                brick_size=self.brick_size,
                trend_direction=trend['direction'],
                brick_count=len(renko_data),
                reversal_points=self._find_reversal_points(renko_data)
            )
        }
