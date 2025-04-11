import numpy as np
from typing import Dict, List
from sklearn.cluster import DBSCAN

class VolumeClusterAnalyzer:
    def __init__(self, eps: float = 0.1, min_samples: int = 5):
        self.clustering = DBSCAN(eps=eps, min_samples=min_samples)
        
    async def analyze_clusters(self, volume_data: np.ndarray) -> Dict:
        """거래량 클러스터 분석"""
        clusters = self.clustering.fit_predict(volume_data.reshape(-1, 1))
        
        return {
            'cluster_centers': self._find_cluster_centers(volume_data, clusters),
            'cluster_sizes': self._calculate_cluster_sizes(clusters),
            'significance': self._evaluate_cluster_significance(clusters)
        }
