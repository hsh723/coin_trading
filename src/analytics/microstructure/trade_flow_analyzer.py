import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class TradeFlowMetrics:
    flow_imbalance: float
    trade_intensity: float
    average_trade_size: float
    trade_direction: str

class TradeFlowAnalyzer:
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        
    async def analyze_flow(self, trades: List[Dict]) -> TradeFlowMetrics:
        """거래 흐름 분석"""
        recent_trades = trades[-self.window_size:]
        imbalance = self._calculate_imbalance(recent_trades)
        
        return TradeFlowMetrics(
            flow_imbalance=imbalance,
            trade_intensity=self._calculate_intensity(recent_trades),
            average_trade_size=np.mean([t['size'] for t in recent_trades]),
            trade_direction='buy' if imbalance > 0 else 'sell'
        )
