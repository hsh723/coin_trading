from typing import Dict, List
from dataclasses import dataclass
from sklearn.cluster import KMeans
import numpy as np

@dataclass
class ClusterSignal:
    cluster_id: int
    cluster_center: float
    distance_to_center: float
    regime_type: str
    confidence: float

class ClusteringStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'n_clusters': 3,
            'features': ['price', 'volume', 'volatility'],
            'lookback': 100
        }
        self.model = KMeans(n_clusters=self.config['n_clusters'])
        
    async def analyze_clusters(self, market_data: pd.DataFrame) -> ClusterSignal:
        """클러스터 분석 및 신호 생성"""
        features = self._extract_features(market_data)
        clusters = self.model.fit_predict(features)
        
        current_cluster = clusters[-1]
        center = self.model.cluster_centers_[current_cluster]
        
        return ClusterSignal(
            cluster_id=current_cluster,
            cluster_center=center[0],  # price center
            distance_to_center=self._calculate_distance(features[-1], center),
            regime_type=self._identify_regime(current_cluster),
            confidence=self._calculate_confidence(features[-1], center)
        )
