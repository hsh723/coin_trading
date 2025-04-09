from typing import Dict, List
import numpy as np
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    position_risk: float
    leverage_risk: float
    concentration_risk: float
    total_risk: float

class PositionRiskManager:
    def __init__(self, risk_limits: Dict = None):
        self.risk_limits = risk_limits or {
            'max_position_size': 0.1,
            'max_leverage': 3.0,
            'max_concentration': 0.3
        }
        
    def calculate_position_risk(self, positions: Dict, 
                              market_data: Dict) -> Dict[str, RiskMetrics]:
        """포지션 리스크 계산"""
        risk_metrics = {}
        for symbol, position in positions.items():
            metrics = self._calculate_risk_metrics(position, market_data[symbol])
            risk_metrics[symbol] = metrics
            
        return risk_metrics
