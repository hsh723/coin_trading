import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from dataclasses import dataclass

@dataclass
class DecompositionResult:
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    strength: float

class TimeSeriesDecomposer:
    def __init__(self, period: int = 24):
        self.period = period
        
    def decompose_series(self, data: pd.Series) -> DecompositionResult:
        """시계열 데이터 분해 분석"""
        decomposition = seasonal_decompose(
            data,
            period=self.period,
            extrapolate_trend='freq'
        )
        
        return DecompositionResult(
            trend=decomposition.trend,
            seasonal=decomposition.seasonal,
            residual=decomposition.resid,
            strength=self._calculate_seasonal_strength(decomposition)
        )
