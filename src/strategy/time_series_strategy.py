from typing import Dict, List
from dataclasses import dataclass
import statsmodels.api as sm
import numpy as np

@dataclass
class TimeSeriesSignal:
    trend_component: np.ndarray
    seasonal_component: np.ndarray
    residual_component: np.ndarray
    forecast: Dict[str, float]
    confidence_intervals: Dict[str, List[float]]

class TimeSeriesStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'decomposition_method': 'multiplicative',
            'forecast_periods': 10,
            'confidence_level': 0.95
        }
        
    async def analyze_timeseries(self, price_data: pd.Series) -> TimeSeriesSignal:
        """시계열 분석 및 예측"""
        # 시계열 분해
        decomposition = sm.tsa.seasonal_decompose(
            price_data,
            model=self.config['decomposition_method']
        )
        
        # SARIMA 모델 적합 및 예측
        model = self._fit_sarima_model(price_data)
        forecast = self._generate_forecast(model)
        
        return TimeSeriesSignal(
            trend_component=decomposition.trend,
            seasonal_component=decomposition.seasonal,
            residual_component=decomposition.resid,
            forecast=forecast,
            confidence_intervals=self._calculate_confidence_intervals(model)
        )
