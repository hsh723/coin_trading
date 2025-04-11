import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class OrderSimulationResult:
    expected_cost: float
    price_impact: float
    optimal_size: float
    execution_schedule: List[Dict]

class OrderSimulator:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'market_impact_factor': 0.1,
            'min_trade_size': 0.01
        }
        
    async def simulate_order(self, order_size: float, market_data: Dict) -> OrderSimulationResult:
        """주문 시뮬레이션"""
        impact = self._estimate_price_impact(order_size, market_data)
        optimal_size = self._calculate_optimal_size(order_size, impact)
        schedule = self._create_execution_schedule(optimal_size)
        
        return OrderSimulationResult(
            expected_cost=self._calculate_expected_cost(impact, optimal_size),
            price_impact=impact,
            optimal_size=optimal_size,
            execution_schedule=schedule
        )
