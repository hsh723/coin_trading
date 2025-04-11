from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class VolumeCluster:
    price_level: float
    volume_weight: float
    cluster_strength: float
    support_resistance: bool

class ClusterVolumeStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_cluster_size': 100,
            'volume_threshold': 0.1,
            'price_variance': 0.002
        }
        
    async def analyze_clusters(self, market_data: pd.DataFrame) -> List[VolumeCluster]:
        """거래량 클러스터 분석"""
        clusters = []
        price_points = self._identify_price_points(market_data)
        
        for price in price_points:
            volume_weight = self._calculate_volume_weight(market_data, price)
            if volume_weight > self.config['volume_threshold']:
                clusters.append(
                    VolumeCluster(
                        price_level=price,
                        volume_weight=volume_weight,
                        cluster_strength=self._calculate_strength(volume_weight),
                        support_resistance=self._is_support_resistance(price, market_data)
                    )
                )
                
        return sorted(clusters, key=lambda x: x.volume_weight, reverse=True)
