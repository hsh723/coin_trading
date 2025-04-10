from typing import Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class OrderSize:
    base_size: float
    adjusted_size: float
    leverage: float
    margin_required: float

class OrderSizer:
    def __init__(self, sizing_config: Dict = None):
        self.config = sizing_config or {
            'base_size': 0.01,  # 1%
            'max_size': 0.1,    # 10%
            'position_scaling': True
        }
        
    async def calculate_order_size(self, 
                                 capital: float, 
                                 risk_metrics: Dict) -> OrderSize:
        """주문 크기 계산"""
        base_size = capital * self.config['base_size']
        adjusted_size = self._adjust_for_risk(base_size, risk_metrics)
        leverage = self._calculate_optimal_leverage(risk_metrics)
        
        return OrderSize(
            base_size=base_size,
            adjusted_size=min(adjusted_size, capital * self.config['max_size']),
            leverage=leverage,
            margin_required=self._calculate_margin_requirement(adjusted_size, leverage)
        )
