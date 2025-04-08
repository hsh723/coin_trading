import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Dict

class MarketRegimeAnalyzer:
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = GaussianMixture(n_components=n_regimes)
        
    def identify_regime(self, returns: pd.Series, volatility: pd.Series) -> str:
        """현재 시장 국면 식별"""
        features = np.column_stack([returns, volatility])
        labels = self.model.fit_predict(features)
        current_regime = labels[-1]
        
        regimes = {
            0: "LOW_VOLATILITY",
            1: "NORMAL",
            2: "HIGH_VOLATILITY"
        }
        return regimes.get(current_regime, "UNKNOWN")
