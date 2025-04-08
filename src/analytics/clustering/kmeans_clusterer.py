from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import List, Dict

class MarketClusterer:
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42
        )
        
    def analyze_clusters(self, market_data: pd.DataFrame) -> Dict:
        """시장 상태 클러스터링"""
        features = self._extract_features(market_data)
        scaled_features = self.scaler.fit_transform(features)
        labels = self.model.fit_predict(scaled_features)
        
        return {
            'cluster_labels': labels,
            'cluster_centers': self.model.cluster_centers_,
            'current_cluster': labels[-1],
            'cluster_stats': self._calculate_cluster_stats(features, labels)
        }
