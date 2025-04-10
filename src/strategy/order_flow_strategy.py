from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class OrderFlowSignal:
    imbalance_ratio: float
    buy_pressure: float
    sell_pressure: float
    signal_type: str
    strength: float

class OrderFlowStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'imbalance_threshold': 0.6,
            'volume_window': 20,
            'pressure_threshold': 0.7
        }
        
    async def analyze_order_flow(self, market_data: pd.DataFrame) -> OrderFlowSignal:
        """주문 흐름 분석"""
        buy_volume = self._calculate_buy_volume(market_data)
        sell_volume = self._calculate_sell_volume(market_data)
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        return OrderFlowSignal(
            imbalance_ratio=imbalance,
            buy_pressure=buy_volume / (buy_volume + sell_volume),
            sell_pressure=sell_volume / (buy_volume + sell_volume),
            signal_type=self._determine_signal(imbalance),
            strength=abs(imbalance)
        )
