import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Dict

class FactorAnalyzer:
    def __init__(self, n_factors: int = 3):
        self.n_factors = n_factors
        self.pca = PCA(n_components=n_factors)
        
    def analyze_risk_factors(self, returns: pd.DataFrame) -> Dict:
        """주요 위험 요인 분석"""
        factors = self.pca.fit_transform(returns)
        exposures = self.pca.components_
        
        return {
            'factors': pd.DataFrame(factors, index=returns.index),
            'exposures': pd.DataFrame(exposures, columns=returns.columns),
            'explained_variance': self.pca.explained_variance_ratio_
        }
