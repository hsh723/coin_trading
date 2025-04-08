import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class CustomIndicator:
    name: str
    value: float
    signal: str
    parameters: Dict

class CustomIndicatorCalculator:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def calculate_custom_ma(self, data: pd.Series, periods: List[int]) -> Dict[str, pd.Series]:
        """사용자 정의 이동평균 계산"""
        results = {}
        for period in periods:
            ma = data.ewm(span=period, adjust=False).mean()
            results[f'EMA_{period}'] = ma
        return results

    def calculate_wave_trend(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Wave Trend 지표 계산"""
        ap = (high + low + close) / 3
        esa = ap.ewm(span=10).mean()
        d = pd.Series(abs(ap - esa)).ewm(span=10).mean()
        ci = (ap - esa) / (0.015 * d)
        return ci.ewm(span=21).mean()
