import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class CorrelationMetrics:
    matrix: pd.DataFrame
    high_correlation_pairs: List[tuple]
    risk_contribution: Dict[str, float]

class CorrelationMatrixAnalyzer:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        
    def analyze_correlations(self, returns: pd.DataFrame) -> CorrelationMetrics:
        """상관관계 분석 수행"""
        corr_matrix = returns.corr()
        high_corr = self._find_high_correlations(corr_matrix)
        risk_contrib = self._calculate_risk_contribution(returns)
        
        return CorrelationMetrics(
            matrix=corr_matrix,
            high_correlation_pairs=high_corr,
            risk_contribution=risk_contrib
        )
