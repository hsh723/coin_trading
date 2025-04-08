import numpy as np
import pandas as pd
from typing import Dict

class RiskAttributionAnalyzer:
    def __init__(self):
        self.risk_metrics = {}
        
    def calculate_risk_contributions(self, returns: pd.DataFrame, weights: np.ndarray) -> Dict:
        """포트폴리오 위험 기여도 분석"""
        cov_matrix = returns.cov()
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # 한계 위험 기여도 계산
        mrc = np.dot(cov_matrix, weights) / portfolio_vol
        # 구성 요소별 위험 기여도
        rc = weights * mrc
