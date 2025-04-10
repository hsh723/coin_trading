from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class PriceLevels:
    support_levels: List[float]
    resistance_levels: List[float]
    strength_scores: Dict[float, float]
    breakout_zones: List[Dict[str, float]]

class SupportResistanceAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'min_touches': 3,
            'price_threshold': 0.02,
            'strength_window': 50
        }
        
    async def find_levels(self, price_data: pd.DataFrame) -> PriceLevels:
        """지지/저항 레벨 탐색"""
        pivots = self._find_pivot_points(price_data)
        clusters = self._cluster_price_levels(pivots)
        strengths = self._calculate_level_strength(clusters, price_data)
        
        return PriceLevels(
            support_levels=self._identify_supports(clusters),
            resistance_levels=self._identify_resistances(clusters),
            strength_scores=strengths,
            breakout_zones=self._find_breakout_zones(clusters, price_data)
        )
