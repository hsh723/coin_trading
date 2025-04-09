from typing import Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class MarginRequirements:
    initial_margin: float
    maintenance_margin: float
    leverage: float
    liquidation_price: float

class MarginCalculator:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'maintenance_margin_ratio': 0.005,
            'initial_margin_ratio': 0.01,
            'max_leverage': 20
        }
        
    async def calculate_margin_requirements(self, 
                                         position: Dict, 
                                         market_price: float) -> MarginRequirements:
        """마진 요구사항 계산"""
        position_value = position['size'] * market_price
        leverage = position['leverage']
        
        initial_margin = position_value / leverage
        maintenance_margin = initial_margin * self.config['maintenance_margin_ratio']
        liquidation_price = self._calculate_liquidation_price(
            position, market_price, maintenance_margin
        )
        
        return MarginRequirements(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            leverage=leverage,
            liquidation_price=liquidation_price
        )
