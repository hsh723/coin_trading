import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class HedgeMetrics:
    hedge_ratio: float
    correlation: float
    beta: float
    hedge_effectiveness: float

class HedgingAnalyzer:
    def __init__(self, lookback_period: int = 30):
        self.lookback_period = lookback_period
        
    def calculate_hedge_ratios(self, asset_returns: pd.Series, 
                             hedge_returns: pd.Series) -> HedgeMetrics:
        """헤지 비율 계산"""
        # 베타 계산
        covariance = asset_returns.cov(hedge_returns)
        hedge_variance = hedge_returns.var()
        beta = covariance / hedge_variance
        
        # 상관관계 계산
        correlation = asset_returns.corr(hedge_returns)
        
        return HedgeMetrics(
            hedge_ratio=beta,
            correlation=correlation,
            beta=beta,
            hedge_effectiveness=correlation ** 2
        )
