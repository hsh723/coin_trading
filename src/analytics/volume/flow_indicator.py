import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class VolumeFlow:
    net_flow: float
    flow_strength: float
    buying_pressure: float
    selling_pressure: float

class VolumeFlowIndicator:
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.flow_history = []
        
    async def analyze_flow(self, trade_data: Dict) -> VolumeFlow:
        """거래량 흐름 분석"""
        buy_volume = self._calculate_buy_volume(trade_data)
        sell_volume = self._calculate_sell_volume(trade_data)
        net_flow = buy_volume - sell_volume
        
        return VolumeFlow(
            net_flow=net_flow,
            flow_strength=abs(net_flow) / (buy_volume + sell_volume),
            buying_pressure=buy_volume / (buy_volume + sell_volume),
            selling_pressure=sell_volume / (buy_volume + sell_volume)
        )
