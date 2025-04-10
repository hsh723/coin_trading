from typing import Dict, List
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import numpy as np

@dataclass
class ClusteringResult:
    pattern_groups: Dict[int, List[float]]
    noise_points: List[float]
    cluster_stats: Dict[str, float]
    dominant_pattern: int

class PatternClusterAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'eps': 0.05,
            'min_samples': 5,
            'pattern_length': 20
        }
        self.model = DBSCAN(
            eps=self.config['eps'],
            min_samples=self.config['min_samples']
        )
        
    async def analyze_patterns(self, market_data: pd.DataFrame) -> ClusteringResult:
        """가격 패턴 클러스터링 분석"""
        patterns = self._extract_patterns(market_data)
        clusters = self.model.fit_predict(patterns)
        
        return ClusteringResult(
            pattern_groups=self._group_patterns(patterns, clusters),
            noise_points=self._identify_noise(patterns, clusters),
            cluster_stats=self._calculate_cluster_stats(clusters),
            dominant_pattern=self._find_dominant_pattern(clusters)
        )
