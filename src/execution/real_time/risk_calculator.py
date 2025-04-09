import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class RealTimeRisk:
    var_95: float
    expected_shortfall: float
    leverage_ratio: float
    margin_usage: float

class RealTimeRiskCalculator:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'var_window': 100,
            'confidence_level': 0.95
        }
        
    def calculate_real_time_risk(self, positions: Dict, 
                               market_data: Dict) -> RealTimeRisk:
        """실시간 리스크 계산"""
        portfolio_value = self._calculate_portfolio_value(positions, market_data)
        var = self._calculate_var(positions, market_data)
        es = self._calculate_expected_shortfall(positions, market_data)
        
        return RealTimeRisk(
            var_95=var,
            expected_shortfall=es,
            leverage_ratio=self._calculate_leverage(positions, portfolio_value),
            margin_usage=self._calculate_margin_usage(positions)
        )
