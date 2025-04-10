from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class TapeReadingSignal:
    order_flow_imbalance: float
    large_orders: List[Dict]
    price_impact: float
    signal_type: str

class TapeReadingStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'large_order_threshold': 1.0,  # BTC
            'impact_window': 100,
            'imbalance_threshold': 0.6
        }
        
    async def analyze_tape(self, order_flow: pd.DataFrame) -> TapeReadingSignal:
        """테이프 리딩 분석"""
        large_orders = self._identify_large_orders(order_flow)
        imbalance = self._calculate_order_imbalance(order_flow)
        impact = self._estimate_price_impact(large_orders)
        
        return TapeReadingSignal(
            order_flow_imbalance=imbalance,
            large_orders=large_orders,
            price_impact=impact,
            signal_type=self._determine_signal(imbalance, impact)
        )
