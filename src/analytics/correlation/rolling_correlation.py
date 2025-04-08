import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class CorrelationMetrics:
    current_correlation: float
    correlation_trend: str
    stability_score: float
    regimes: Dict[str, List[tuple]]

class RollingCorrelationAnalyzer:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        
    def analyze_correlation(self, series1: pd.Series, 
                          series2: pd.Series) -> CorrelationMetrics:
        """동적 상관관계 분석"""
        rolling_corr = self._calculate_rolling_correlation(series1, series2)
        current_corr = rolling_corr.iloc[-1]
        
        return CorrelationMetrics(
            current_correlation=current_corr,
            correlation_trend=self._determine_correlation_trend(rolling_corr),
            stability_score=self._calculate_stability(rolling_corr),
            regimes=self._identify_correlation_regimes(rolling_corr)
        )
