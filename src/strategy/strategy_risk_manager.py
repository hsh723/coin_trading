from typing import Dict, List
from dataclasses import dataclass

@dataclass
class RiskAssessment:
    strategy_id: str
    risk_level: float
    exposure_limits: Dict[str, float]
    stop_levels: Dict[str, float]

class StrategyRiskManager:
    def __init__(self, risk_config: Dict = None):
        self.config = risk_config or {
            'max_drawdown': 0.1,
            'position_limit': 0.2,
            'stop_multiplier': 2.0
        }
        
    async def assess_strategy_risk(self, 
                                 strategy_id: str, 
                                 portfolio_state: Dict) -> RiskAssessment:
        """전략별 리스크 평가"""
        current_drawdown = self._calculate_drawdown(portfolio_state)
        exposure = self._calculate_exposure(portfolio_state)
        
        risk_level = max(
            current_drawdown / self.config['max_drawdown'],
            exposure / self.config['position_limit']
        )
        
        return RiskAssessment(
            strategy_id=strategy_id,
            risk_level=risk_level,
            exposure_limits=self._calculate_exposure_limits(risk_level),
            stop_levels=self._calculate_stop_levels(portfolio_state)
        )
