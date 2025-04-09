from typing import Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class ExecutionMetrics:
    fill_rate: float
    execution_speed: float
    price_improvement: float
    implementation_shortfall: float
    timing_cost: float

class ExecutionPerformanceAnalyzer:
    def __init__(self):
        self.metrics_history = []
        
    def calculate_metrics(self, order: Dict, execution: Dict) -> ExecutionMetrics:
        """실행 성능 지표 계산"""
        fill_rate = execution['filled_amount'] / order['amount']
        execution_time = (execution['end_time'] - execution['start_time']).total_seconds()
        
        return ExecutionMetrics(
            fill_rate=fill_rate,
            execution_speed=execution['filled_amount'] / execution_time,
            price_improvement=self._calculate_price_improvement(order, execution),
            implementation_shortfall=self._calculate_shortfall(order, execution),
            timing_cost=self._calculate_timing_cost(order, execution)
        )
