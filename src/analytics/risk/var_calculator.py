import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

class VaRCalculator:
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
    def calculate_var(self, returns: pd.Series, position_value: float) -> Dict[str, float]:
        """VaR 계산"""
        # 정규분포 기반 VaR
        parametric_var = self._calculate_parametric_var(returns, position_value)
        
        # 히스토리컬 VaR
        historical_var = self._calculate_historical_var(returns, position_value)
        
        # 조건부 VaR (Expected Shortfall)
        cvar = self._calculate_cvar(returns, position_value)
        
        return {
            'parametric_var': parametric_var,
            'historical_var': historical_var,
            'cvar': cvar
        }
