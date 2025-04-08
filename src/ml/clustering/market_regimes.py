import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import List, Dict

class MarketRegimeClusterer:
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.model = KMeans(n_clusters=n_regimes)
        self.regime_labels = ['LOW_VOL', 'NORMAL', 'HIGH_VOL', 'CRISIS']
        
    def identify_regime(self, market_features: pd.DataFrame) -> str:
        """현재 시장 국면 식별"""
        labels = self.model.fit_predict(market_features)
        current_regime = labels[-1]
        
        # 군집 특성 분석하여 레이블 매핑
        cluster_centers = self.model.cluster_centers_
        volatility_ranking = np.argsort(cluster_centers[:, 0])
        regime_mapping = dict(zip(volatility_ranking, self.regime_labels))
        
        return regime_mapping[current_regime]
