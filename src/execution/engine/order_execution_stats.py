from typing import Dict, List
from dataclasses import dataclass
import time

@dataclass
class ExecutionStats:
    order_count: int
    success_rate: float
    avg_execution_time: float
    fill_rates: Dict[str, float]
    cost_analysis: Dict[str, float]

class OrderExecutionStats:
    def __init__(self, stats_window: int = 1000):
        self.stats_window = stats_window
        self.executions = []
        
    async def update_stats(self, execution_result: Dict) -> ExecutionStats:
        """실행 통계 업데이트"""
        self.executions.append(execution_result)
        if len(self.executions) > self.stats_window:
            self.executions.pop(0)
            
        return ExecutionStats(
            order_count=len(self.executions),
            success_rate=self._calculate_success_rate(),
            avg_execution_time=self._calculate_avg_execution_time(),
            fill_rates=self._analyze_fill_rates(),
            cost_analysis=self._analyze_execution_costs()
        )
