from typing import Dict, List
from dataclasses import dataclass
import numpy as np
from statsmodels.tsa.stattools import adfuller

@dataclass
class TimeSeriesAnalysis:
    is_stationary: bool
    seasonality_info: Dict[str, float]
    trend_components: Dict[str, np.ndarray]
    cycle_length: int

class TimeSeriesAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'decomposition_method': 'multiplicative',
            'cycle_detection_window': 30
        }
        
    async def analyze_timeseries(self, price_data: pd.Series) -> TimeSeriesAnalysis:
        """시계열 분석 수행"""
        return TimeSeriesAnalysis(
            is_stationary=self._check_stationarity(price_data),
            seasonality_info=self._detect_seasonality(price_data),
            trend_components=self._decompose_trend(price_data),
            cycle_length=self._find_cycle_length(price_data)
        )
