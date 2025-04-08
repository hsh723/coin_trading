import numpy as np
import pandas as pd
from arch import arch_model
from typing import Dict
from dataclasses import dataclass

@dataclass
class VolatilityForecast:
    point_estimate: float
    confidence_interval: tuple
    regime: str
    forecast_horizon: int

class VolatilityForecaster:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'model_type': 'GARCH',
            'horizon': 5,
            'p': 1,
            'q': 1
        }
        
    def forecast_volatility(self, returns: pd.Series) -> VolatilityForecast:
        """GARCH 모델을 사용한 변동성 예측"""
        model = arch_model(
            returns,
            vol='Garch',
            p=self.config['p'],
            q=self.config['q']
        )
        res = model.fit(disp='off')
        forecast = res.forecast(horizon=self.config['horizon'])
        
        return VolatilityForecast(
            point_estimate=np.sqrt(forecast.variance.mean().iloc[-1]),
            confidence_interval=self._calculate_confidence_interval(forecast),
            regime=self._determine_regime(forecast),
            forecast_horizon=self.config['horizon']
        )
