from dataclasses import dataclass
from typing import Dict
import pandas as pd

@dataclass
class ExecutionQuality:
    price_improvement: float
    execution_speed: float
    fill_ratio: float
    cost_savings: float

class ExecutionQualityMonitor:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'speed_threshold': 1.0,  # 초
            'fill_ratio_min': 0.95
        }
        
    async def monitor_execution(self, order: Dict, execution: Dict) -> ExecutionQuality:
        """실행 품질 모니터링"""
        price_imp = self._calculate_price_improvement(order, execution)
        speed = self._calculate_execution_speed(execution)
        
        return ExecutionQuality(
            price_improvement=price_imp,
            execution_speed=speed,
            fill_ratio=execution['filled'] / order['amount'],
            cost_savings=self._calculate_cost_savings(price_imp, execution)
        )
