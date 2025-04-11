import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolumeLevels:
    support_levels: List[float]
    resistance_levels: List[float]
    volume_clusters: List[Dict[str, float]]
    strength_scores: Dict[float, float]

class VolumeSupportResistance:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'cluster_threshold': 0.1,
            'min_volume_ratio': 0.02
        }
        
    async def find_volume_levels(self, price_data: np.ndarray, volume_data: np.ndarray) -> VolumeLevels:
        """거래량 기반 지지/저항 레벨 분석"""
        clusters = self._find_volume_clusters(price_data, volume_data)
        support, resistance = self._classify_levels(clusters)
        
        return VolumeLevels(
            support_levels=support,
            resistance_levels=resistance,
            volume_clusters=clusters,
            strength_scores=self._calculate_level_strength(clusters)
        )
