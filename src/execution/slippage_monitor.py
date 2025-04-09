import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class SlippageMetrics:
    realized_slippage: float
    expected_slippage: float
    deviation: float
    impact_score: float

class SlippageMonitor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'acceptable_deviation': 0.001,
            'impact_window': 20
        }
        
    async def monitor_slippage(self, order: Dict, execution: Dict) -> SlippageMetrics:
        """실행 슬리피지 모니터링"""
        expected_price = order['price']
        executed_price = execution['average_price']
        realized_slip = (executed_price - expected_price) / expected_price
        
        return SlippageMetrics(
            realized_slippage=realized_slip,
            expected_slippage=self._calculate_expected_slippage(order),
            deviation=abs(realized_slip),
            impact_score=self._calculate_impact_score(order, execution)
        )
