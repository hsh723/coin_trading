from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class ImbalanceMetrics:
    buy_sell_ratio: float
    imbalance_score: float
    pressure_direction: str
    significant_levels: List[float]

class OrderImbalanceAnalyzer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.order_history = []
        
    async def analyze_imbalance(self, order_book: Dict) -> ImbalanceMetrics:
        """실시간 주문 불균형 분석"""
        buy_volume = sum(level['volume'] for level in order_book['bids'])
        sell_volume = sum(level['volume'] for level in order_book['asks'])
        
        ratio = buy_volume / sell_volume if sell_volume > 0 else float('inf')
        score = self._calculate_imbalance_score(buy_volume, sell_volume)
        
        return ImbalanceMetrics(
            buy_sell_ratio=ratio,
            imbalance_score=score,
            pressure_direction=self._determine_pressure(ratio),
            significant_levels=self._find_significant_levels(order_book)
        )
