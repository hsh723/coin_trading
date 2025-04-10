from typing import Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class PositionSize:
    amount: float
    leverage: float
    risk_amount: float
    max_loss: float

class PositionCalculator:
    def __init__(self, calculator_config: Dict = None):
        self.config = calculator_config or {
            'max_position_size': 0.1,  # 전체 자본의 10%
            'max_leverage': 5,
            'risk_per_trade': 0.01  # 1%
        }
        
    async def calculate_position_size(self, 
                                   capital: float,
                                   entry_price: float,
                                   stop_loss: float) -> PositionSize:
        """포지션 크기 계산"""
        risk_amount = capital * self.config['risk_per_trade']
        price_risk = abs(entry_price - stop_loss) / entry_price
        position_size = risk_amount / price_risk
        
        return PositionSize(
            amount=min(position_size, capital * self.config['max_position_size']),
            leverage=self._calculate_optimal_leverage(price_risk),
            risk_amount=risk_amount,
            max_loss=risk_amount
        )
