from typing import Dict
from dataclasses import dataclass
import pandas as pd

@dataclass
class TradeFlowMetrics:
    buy_pressure: float
    sell_pressure: float
    net_flow: float
    trade_imbalance: float
    flow_strength: str

class TradeFlowAnalyzer:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.trade_history = []
        
    async def analyze_trade_flow(self, recent_trades: List[Dict]) -> TradeFlowMetrics:
        """실시간 거래 흐름 분석"""
        buy_volume = sum(t['volume'] for t in recent_trades if t['side'] == 'buy')
        sell_volume = sum(t['volume'] for t in recent_trades if t['side'] == 'sell')
        
        return TradeFlowMetrics(
            buy_pressure=self._calculate_buy_pressure(recent_trades),
            sell_pressure=self._calculate_sell_pressure(recent_trades),
            net_flow=buy_volume - sell_volume,
            trade_imbalance=(buy_volume - sell_volume)/(buy_volume + sell_volume),
            flow_strength=self._determine_flow_strength(buy_volume, sell_volume)
        )
