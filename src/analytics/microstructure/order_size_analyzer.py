import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class OrderSizeAnalysis:
    average_size: float
    size_distribution: Dict[str, float]
    large_orders: List[Dict]
    size_impact: float

class OrderSizeAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'large_order_threshold': 2.0,
            'impact_window': 50
        }
        
    async def analyze_order_sizes(self, trade_data: List[Dict]) -> OrderSizeAnalysis:
        """주문 크기 분석"""
        sizes = [trade['size'] for trade in trade_data]
        avg_size = np.mean(sizes)
        
        return OrderSizeAnalysis(
            average_size=avg_size,
            size_distribution=self._calculate_size_distribution(sizes),
            large_orders=self._identify_large_orders(trade_data),
            size_impact=self._calculate_size_impact(trade_data)
        )
