import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PositionMetrics:
    exposure: float
    risk_contribution: float
    profit_loss: float
    holding_period: int
    cost_basis: float

class PositionAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
    def analyze_position(self, position_data: Dict, 
                        market_data: pd.DataFrame) -> PositionMetrics:
        """포지션 상세 분석"""
        current_price = market_data['close'].iloc[-1]
        entry_price = position_data['entry_price']
        size = position_data['size']
        
        exposure = size * current_price
        profit_loss = (current_price - entry_price) * size
        
        return PositionMetrics(
            exposure=exposure,
            risk_contribution=self._calculate_risk_contribution(position_data, market_data),
            profit_loss=profit_loss,
            holding_period=self._calculate_holding_period(position_data),
            cost_basis=entry_price
        )
