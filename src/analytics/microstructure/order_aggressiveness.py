import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class AggressivenessMetrics:
    aggression_score: float
    market_impact: float
    order_flow_toxicity: float
    trader_behavior: str

class OrderAggressivenessAnalyzer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
    async def analyze_aggressiveness(self, order_flow: Dict) -> AggressivenessMetrics:
        """주문 공격성 분석"""
        return AggressivenessMetrics(
            aggression_score=self._calculate_aggression_score(order_flow),
            market_impact=self._estimate_market_impact(order_flow),
            order_flow_toxicity=self._calculate_toxicity(order_flow),
            trader_behavior=self._classify_behavior(order_flow)
        )
