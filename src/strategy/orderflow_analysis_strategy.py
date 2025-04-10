from typing import Dict, List
from dataclasses import dataclass
import pandas as pd

@dataclass
class OrderFlowAnalysis:
    buy_pressure: float
    sell_pressure: float
    trade_flow: Dict[str, float]
    market_depth: Dict[str, float]
    signal_strength: float

class OrderFlowAnalysisStrategy:
    def __init__(self, config: Dict = None):
        self.config = config or {
            'depth_levels': 10,
            'flow_window': 50,
            'volume_threshold': 1.0
        }
        
    async def analyze_orderflow(self, order_data: pd.DataFrame) -> OrderFlowAnalysis:
        """주문 흐름 분석 실행"""
        trade_flow = self._calculate_trade_flow(order_data)
        depth = self._analyze_market_depth(order_data)
        
        return OrderFlowAnalysis(
            buy_pressure=self._calculate_buy_pressure(trade_flow),
            sell_pressure=self._calculate_sell_pressure(trade_flow),
            trade_flow=trade_flow,
            market_depth=depth,
            signal_strength=self._calculate_signal_strength(trade_flow, depth)
        )
