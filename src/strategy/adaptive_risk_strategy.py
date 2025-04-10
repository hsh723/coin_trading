from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class RiskAdjustment:
    position_size: float
    stop_loss: float
    take_profit: float
    risk_ratio: float

class AdaptiveRiskStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'base_risk': 0.02,  # 2% 기본 리스크
            'volatility_lookback': 20,
            'max_position_size': 0.1  # 10% 최대 포지션
        }
        
    async def calculate_risk_adjustment(self, 
                                     market_data: pd.DataFrame) -> RiskAdjustment:
        """적응형 리스크 계산"""
        volatility = self._calculate_volatility(market_data)
        risk_factor = self._adjust_risk_for_volatility(volatility)
        
        return RiskAdjustment(
            position_size=self._calculate_position_size(risk_factor),
            stop_loss=self._calculate_stop_loss(volatility),
            take_profit=self._calculate_take_profit(volatility),
            risk_ratio=risk_factor
        )
