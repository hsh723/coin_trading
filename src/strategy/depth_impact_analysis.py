from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class DepthImpactMetrics:
    bid_depth: Dict[float, float]
    ask_depth: Dict[float, float]
    impact_coefficient: float
    slippage_estimate: Dict[str, float]

class DepthImpactAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'depth_levels': 20,
            'impact_threshold': 0.005,
            'size_increments': [0.1, 0.5, 1.0, 2.0, 5.0]
        }
        
    async def analyze_depth_impact(self, order_book: pd.DataFrame) -> DepthImpactMetrics:
        """주문장 깊이 영향도 분석"""
        bid_depth = self._calculate_cumulative_depth(order_book['bids'])
        ask_depth = self._calculate_cumulative_depth(order_book['asks'])
        
        return DepthImpactMetrics(
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            impact_coefficient=self._calculate_impact_coefficient(bid_depth, ask_depth),
            slippage_estimate=self._estimate_slippage_costs(bid_depth, ask_depth)
        )
