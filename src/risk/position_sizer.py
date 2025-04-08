from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class PositionSize:
    amount: float
    max_loss: float
    risk_percentage: float

class PositionSizer:
    def __init__(self, config: Dict):
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.02)
        self.max_position_size = config.get('max_position_size', 0.1)
        self.account_balance = config.get('initial_balance', 10000)
        
    def calculate_position_size(self, 
                              entry_price: float,
                              stop_loss: float,
                              volatility: float = None) -> PositionSize:
        """포지션 크기 계산"""
        risk_amount = self.account_balance * self.max_risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        base_size = risk_amount / risk_per_unit
        if volatility:
            base_size *= self._adjust_for_volatility(volatility)
