from typing import Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class PositionSize:
    base_size: float
    adjusted_size: float
    risk_allocation: float
    max_size: float

class DynamicPositionSizer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'base_risk': 0.02,
            'max_position': 0.1,
            'volatility_scale': True
        }
        
    async def calculate_position_size(self, 
                                    capital: float, 
                                    volatility: float) -> PositionSize:
        base = capital * self.config['base_risk']
        adjusted = self._adjust_for_volatility(base, volatility)
        
        return PositionSize(
            base_size=base,
            adjusted_size=min(adjusted, capital * self.config['max_position']),
            risk_allocation=self.config['base_risk'],
            max_size=capital * self.config['max_position']
        )
