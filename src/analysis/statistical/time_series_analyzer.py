import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
from typing import Dict, Tuple

class TimeSeriesAnalyzer:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def analyze_stationarity(self, series: pd.Series) -> Dict:
        """시계열 정상성 분석"""
        adf_result = adfuller(series)
        kpss_result = kpss(series)
        
        return {
            'is_stationary': adf_result[1] < self.significance_level,
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1]
        }
