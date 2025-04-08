from typing import Dict, List
import pandas as pd
from dataclasses import dataclass

@dataclass
class OrderFlowMetrics:
    buy_pressure: float
    sell_pressure: float
    net_flow: float
    trade_imbalance: float
    large_trades: List[Dict]

class OrderFlowAnalyzer:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'large_trade_threshold': 100000,
            'timeframe': '1m'
        }
        
    def analyze_order_flow(self, trades: pd.DataFrame) -> OrderFlowMetrics:
        """주문 흐름 분석"""
        buy_trades = trades[trades['side'] == 'buy']
        sell_trades = trades[trades['side'] == 'sell']
        
        return OrderFlowMetrics(
            buy_pressure=self._calculate_pressure(buy_trades),
            sell_pressure=self._calculate_pressure(sell_trades),
            net_flow=buy_trades['volume'].sum() - sell_trades['volume'].sum(),
            trade_imbalance=self._calculate_imbalance(trades),
            large_trades=self._identify_large_trades(trades)
        )
