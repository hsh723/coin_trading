import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Dict

class MarketRegimeClassifier:
    def __init__(self, n_regimes: int = 3, lookback_period: int = 60):
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.model = GaussianMixture(n_components=n_regimes)
        
    def classify_regime(self, market_data: pd.DataFrame) -> str:
        """현재 시장 국면 분류"""
        features = self._extract_regime_features(market_data)
        regime = self.model.fit_predict(features)
        return self._interpret_regime(regime[-1])
