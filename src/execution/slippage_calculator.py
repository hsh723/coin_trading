import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class SlippageMetrics:
    expected_slippage: float
    realized_slippage: float
    impact_cost: float

class SlippageCalculator:
    def __init__(self, market_impact_factor: float = 0.1):
        self.market_impact_factor = market_impact_factor
        
    def calculate_slippage(self, order_size: float, market_data: Dict) -> SlippageMetrics:
        """주문 실행 슬리피지 계산"""
        volume = market_data['volume']
        spread = market_data['ask'] - market_data['bid']
        
        # 시장 영향도 계산
        market_impact = self.market_impact_factor * (order_size / volume) ** 0.5
        
        return SlippageMetrics(
            expected_slippage=spread * 0.5 + market_impact,
            realized_slippage=0.0,  # 실제 실행 후 업데이트
            impact_cost=market_impact
        )
