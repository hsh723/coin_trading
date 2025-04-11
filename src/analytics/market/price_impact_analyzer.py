import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class ImpactAnalysis:
    immediate_impact: float
    expected_slippage: float
    market_resilience: float
    optimal_execution_size: float

class PriceImpactAnalyzer:
    def __init__(self, impact_window: int = 50):
        self.impact_window = impact_window
        
    async def analyze_impact(self, order_book: Dict, trade_size: float) -> ImpactAnalysis:
        """가격 영향도 분석"""
        immediate_impact = self._calculate_immediate_impact(order_book, trade_size)
        slippage = self._estimate_slippage(order_book, trade_size)
        
        return ImpactAnalysis(
            immediate_impact=immediate_impact,
            expected_slippage=slippage,
            market_resilience=self._calculate_resilience(order_book),
            optimal_execution_size=self._calculate_optimal_size(order_book)
        )
