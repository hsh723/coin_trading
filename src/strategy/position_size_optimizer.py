from typing import Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class PositionSizeRecommendation:
    optimal_size: float
    max_size: float
    risk_adjusted_size: float
    kelly_size: float

class PositionSizeOptimizer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_position_size': 0.1,  # 10% of capital
            'risk_per_trade': 0.02,    # 2% risk per trade
            'kelly_fraction': 0.5      # Half-Kelly for safety
        }
        
    async def optimize_position_size(self, 
                                   capital: float,
                                   win_rate: float,
                                   risk_ratio: float) -> PositionSizeRecommendation:
        """포지션 크기 최적화"""
        kelly_size = self._calculate_kelly_criterion(win_rate, risk_ratio)
        max_size = capital * self.config['max_position_size']
        risk_size = capital * self.config['risk_per_trade']
        
        return PositionSizeRecommendation(
            optimal_size=min(kelly_size, max_size),
            max_size=max_size,
            risk_adjusted_size=risk_size,
            kelly_size=kelly_size * self.config['kelly_fraction']
        )
