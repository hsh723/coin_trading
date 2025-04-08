import pandas as pd
import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class VolatilityRegime:
    current_regime: str
    volatility_level: float
    regime_duration: int
    transition_probability: Dict[str, float]

class VolatilityRegimeDetector:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.regimes = ['low', 'medium', 'high', 'extreme']
        
    def detect_regime(self, returns: pd.Series) -> VolatilityRegime:
        """변동성 국면 감지"""
        rolling_vol = returns.rolling(window=self.window_size).std() * np.sqrt(252)
        current_vol = rolling_vol.iloc[-1]
        
        return VolatilityRegime(
            current_regime=self._classify_regime(current_vol),
            volatility_level=current_vol,
            regime_duration=self._calculate_regime_duration(rolling_vol),
            transition_probability=self._calculate_transition_probs(rolling_vol)
        )
