import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class MarketImpactResult:
    price_impact: float
    volume_impact: float
    total_impact_cost: float
    estimated_slippage: float

class MarketImpactCalculator:
    def __init__(self, impact_model: str = 'square_root'):
        self.impact_model = impact_model
        
    def calculate_impact(self, order_size: float, market_data: Dict) -> MarketImpactResult:
        """시장 충격 비용 계산"""
        avg_volume = self._calculate_avg_volume(market_data)
        volume_ratio = order_size / avg_volume
        
        return MarketImpactResult(
            price_impact=self._calculate_price_impact(volume_ratio),
            volume_impact=self._calculate_volume_impact(volume_ratio),
            total_impact_cost=self._calculate_total_impact(volume_ratio),
            estimated_slippage=self._estimate_slippage(volume_ratio)
        )
