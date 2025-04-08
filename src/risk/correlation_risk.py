import numpy as np
import pandas as pd
from typing import Dict, List

class CorrelationRiskManager:
    def __init__(self, max_correlation: float = 0.7):
        self.max_correlation = max_correlation
        self.correlation_matrix = None
        
    def update_correlation_matrix(self, returns: pd.DataFrame):
        """상관관계 매트릭스 업데이트"""
        self.correlation_matrix = returns.corr()
        
    def check_portfolio_correlation(self, assets: List[str]) -> Dict[str, List[str]]:
        """포트폴리오 상관관계 검사"""
        high_correlation_pairs = {}
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                corr = abs(self.correlation_matrix.loc[assets[i], assets[j]])
                if corr > self.max_correlation:
                    high_correlation_pairs.setdefault(assets[i], []).append(assets[j])
        return high_correlation_pairs
