import numpy as np
import pandas as pd
from typing import Dict
from dataclasses import dataclass

@dataclass
class RiskContribution:
    asset_name: str
    volatility: float
    beta: float
    correlation: float
    contribution: float

class RiskAttributionAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02
        
    def analyze_contributions(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict[str, RiskContribution]:
        """포트폴리오 리스크 기여도 분석"""
        cov_matrix = returns.cov()
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        contributions = {}
        for i, asset in enumerate(returns.columns):
            contrib = self._calculate_contribution(
                returns[asset],
                weights[i],
                returns,
                weights,
                cov_matrix
            )
            contributions[asset] = contrib
            
        return contributions
