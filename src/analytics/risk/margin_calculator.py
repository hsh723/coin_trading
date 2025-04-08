from typing import Dict
import numpy as np
from dataclasses import dataclass

@dataclass
class MarginRequirements:
    initial_margin: float
    maintenance_margin: float
    liquidation_price: float
    max_leverage: float

class MarginCalculator:
    def __init__(self, config: Dict):
        self.min_margin_ratio = config.get('min_margin_ratio', 0.05)
        self.margin_multiplier = config.get('margin_multiplier', 1.5)
        
    def calculate_margins(self, 
                        position_size: float,
                        entry_price: float,
                        leverage: float) -> MarginRequirements:
        """마진 요구사항 계산"""
        position_value = position_size * entry_price
        initial_margin = position_value / leverage
        maintenance_margin = initial_margin * self.margin_multiplier
        
        return MarginRequirements(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            liquidation_price=self._calculate_liquidation_price(
                entry_price, leverage, position_size > 0
            ),
            max_leverage=self._calculate_max_leverage(position_value)
        )
