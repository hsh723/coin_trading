from typing import Dict
from dataclasses import dataclass

@dataclass
class MarginRequirements:
    initial_margin: float
    maintenance_margin: float
    leverage: float
    max_position_size: float
    liquidation_price: float

class MarginCalculator:
    def __init__(self, margin_config: Dict = None):
        self.config = margin_config or {
            'initial_margin_ratio': 0.01,  # 1%
            'maintenance_margin_ratio': 0.005,  # 0.5%
            'max_leverage': 100
        }
        
    async def calculate_margin_requirements(self, 
                                         position_size: float,
                                         entry_price: float,
                                         leverage: float) -> MarginRequirements:
        """마진 요구사항 계산"""
        position_value = position_size * entry_price
        
        initial_margin = position_value / leverage
        maintenance_margin = initial_margin * self.config['maintenance_margin_ratio']
        max_position = self._calculate_max_position(leverage, initial_margin)
        
        return MarginRequirements(
            initial_margin=initial_margin,
            maintenance_margin=maintenance_margin,
            leverage=leverage,
            max_position_size=max_position,
            liquidation_price=self._calculate_liquidation_price(
                position_size, entry_price, maintenance_margin
            )
        )
