from typing import Dict
from dataclasses import dataclass

@dataclass
class RiskMetrics:
    position_risk: float
    leverage_risk: float
    margin_risk: float
    liquidation_distance: float
    total_risk_score: float

class RiskCalculator:
    def __init__(self, risk_config: Dict = None):
        self.config = risk_config or {
            'max_position_risk': 0.05,
            'max_leverage': 10.0,
            'min_margin_ratio': 0.05
        }
        
    async def calculate_risk(self, position: Dict, 
                           market_data: Dict) -> RiskMetrics:
        """포지션 리스크 계산"""
        position_risk = self._calculate_position_risk(position, market_data)
        leverage_risk = self._calculate_leverage_risk(position)
        margin_risk = self._calculate_margin_risk(position)
        
        return RiskMetrics(
            position_risk=position_risk,
            leverage_risk=leverage_risk,
            margin_risk=margin_risk,
            liquidation_distance=self._calculate_liquidation_distance(position, market_data),
            total_risk_score=self._calculate_total_risk_score([
                position_risk, leverage_risk, margin_risk
            ])
        )
