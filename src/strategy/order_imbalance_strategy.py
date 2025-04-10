from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class ImbalanceSignal:
    buy_sell_ratio: float
    pressure_direction: str
    imbalance_strength: float
    signal_confidence: float

class OrderImbalanceStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'window_size': 50,
            'imbalance_threshold': 0.6,
            'min_volume': 1.0
        }
        
    async def analyze_imbalance(self, order_flow: pd.DataFrame) -> ImbalanceSignal:
        """주문 불균형 분석"""
        buy_volume = order_flow['buy_volume'].sum()
        sell_volume = order_flow['sell_volume'].sum()
        ratio = buy_volume / (buy_volume + sell_volume)
        
        return ImbalanceSignal(
            buy_sell_ratio=ratio,
            pressure_direction='buy' if ratio > 0.5 else 'sell',
            imbalance_strength=abs(ratio - 0.5) * 2,
            signal_confidence=self._calculate_confidence(ratio)
        )
